package edu.indiana.soic.spidal.damds;

import com.google.common.base.Optional;
import com.google.common.base.Strings;
import edu.indiana.soic.spidal.common.BinaryReader;
import edu.indiana.soic.spidal.common.DoubleStatistics;
import edu.indiana.soic.spidal.common.RefObj;
import edu.indiana.soic.spidal.configuration.ConfigurationMgr;
import edu.indiana.soic.spidal.configuration.section.DAMDSSection;
import mpi.MPIException;
import org.apache.commons.cli.*;

import java.nio.ByteOrder;
import java.util.Date;
import java.util.Random;
import java.util.stream.IntStream;

import static edu.rice.hj.Module0.launchHabaneroApp;
import static edu.rice.hj.Module1.forallChunked;

public class Program {
    private static Options programOptions = new Options();

    static {
        programOptions.addOption(
            String.valueOf(Constants.CMD_OPTION_SHORT_C),
            Constants.CMD_OPTION_LONG_C, true,
            Constants.CMD_OPTION_DESCRIPTION_C);
        programOptions.addOption(
            String.valueOf(Constants.CMD_OPTION_SHORT_N),
            Constants.CMD_OPTION_LONG_N, true,
            Constants.CMD_OPTION_DESCRIPTION_N);
        programOptions.addOption(
            String.valueOf(Constants.CMD_OPTION_SHORT_T),
            Constants.CMD_OPTION_LONG_T, true,
            Constants.CMD_OPTION_DESCRIPTION_T);
    }

    //Config Settings
    public static DAMDSSection config;
    public static ByteOrder byteOrder;
    public static BinaryReader distances;
    public static BinaryReader weights;


    /**
     * Weighted SMACOF based on Deterministic Annealing algorithm
     *
     * @param args command line arguments to the program, which should include
     *             -c path to config file
     *             -t number of threads
     *             -n number of nodes
     *             The options may also be given as longer names
     *             --configFile, --threadCount, and --nodeCount respectively
     */
    public static void main(String[] args) {
        Optional<CommandLine> parserResult =
            parseCommandLineArguments(args, programOptions);

        if (!parserResult.isPresent()) {
            System.out.println(Constants.ERR_PROGRAM_ARGUMENTS_PARSING_FAILED);
            new HelpFormatter()
                .printHelp(Constants.PROGRAM_NAME, programOptions);
            return;
        }

        CommandLine cmd = parserResult.get();
        if (!(cmd.hasOption(Constants.CMD_OPTION_LONG_C) &&
              cmd.hasOption(Constants.CMD_OPTION_LONG_N) &&
              cmd.hasOption(Constants.CMD_OPTION_LONG_T))) {
            System.out.println(Constants.ERR_INVALID_PROGRAM_ARGUMENTS);
            new HelpFormatter()
                .printHelp(Constants.PROGRAM_NAME, programOptions);
            return;
        }

        Utils.printMessage("\n== DAMDS run started on " + new Date() + " ==");

        //  Read Metadata using this as source of other metadata
        readConfiguration(cmd);
        Utils.printMessage(config.toString(true));

        try {
            //  Set up MPI and threads parallelism
            ParallelOps.setupParallelism(args);
            ParallelOps.setParallelDecomposition(config.numberDataPoints);

            readDistancesAndWeights();
            RefObj<Integer> missingDistCount = new RefObj<>();
            DoubleStatistics distanceSummary = calculateStatistics(
                missingDistCount);

            double missingDistPercent = missingDistCount.getValue() /
                                        (Math.pow(config.numberDataPoints, 2));
            Utils.printMessage("\nDistance summary \n" + distanceSummary.toString() +"\n  MissingDistPercentage=" +
                               missingDistPercent);

            double[][] preX = generateInitMapping(
                config.numberDataPoints, config.targetDimension);
            double tCur = 0.0;
            double preStress = calculateStress(preX, tCur, config.targetDimension, config.isSammon, distanceSummary.getAverage());
            Utils.printMessage("\nInitial stress=" + preStress);


            ParallelOps.tearDownParallelism();
        }
        catch (MPIException e) {
            Utils.printAndThrowRuntimeException(new RuntimeException(e));
        }
    }

    private static double calculateStress(double[][] preX, double tCur, int targetDimension, boolean isSammon, double avgDist)

        throws MPIException {
        final double [] sigmaValues = new double [ParallelOps.threadCount];
        IntStream.range(0, ParallelOps.threadCount).forEach(i -> sigmaValues[i] = 0.0);

        if (ParallelOps.threadCount > 1) {
            launchHabaneroApp(
                () -> forallChunked(
                    0, ParallelOps.threadCount - 1,
                    (threadIdx) -> sigmaValues[threadIdx] =
                        calculateStressInternal(threadIdx, preX, targetDimension, tCur, isSammon, avgDist)));
            // Sum across threads and accumulate to zeroth entry
            IntStream.range(1, ParallelOps.threadCount).forEach(
                i -> {
                    sigmaValues[0] += sigmaValues[i];
                });
        }
        else {
            sigmaValues[0] = calculateStressInternal(0, preX, targetDimension, tCur, isSammon, avgDist);
        }

        if (ParallelOps.procCount > 1) {
            sigmaValues[0] = ParallelOps.allReduce(sigmaValues[0]);
        }
        return sigmaValues[0];
    }

    private static double calculateStressInternal(
        int threadIdx, double[][] preX, int targetDim, double tCur,
        boolean isSammon, double avgDist) {

        double sigma = 0.0;
        double diff = 0.0;
        if (tCur > 10E-10) {
            diff = Math.sqrt(2.0 * targetDim) * tCur;
        }

        int pointCount =
            ParallelOps.threadRowCounts[threadIdx] * ParallelOps.globalColCount;

        for (int i = 0; i < pointCount; ++i) {
            int procLocalPnum =
                i + ParallelOps.threadPointStartOffsets[threadIdx];
            double origD = distances.getValue(procLocalPnum);
            double weight = weights.getValue(procLocalPnum);

            if (origD < 0 || weight == 0) {
                continue;
            }

            weight = isSammon ? 1.0 / Math.max(origD, 0.001 * avgDist) : weight;
            int globalPointStart =
                procLocalPnum + ParallelOps.procPointStartOffset;
            int globalRow = globalPointStart / ParallelOps.globalColCount;
            int globalCol = globalPointStart % ParallelOps.globalColCount;

            double euclideanD = globalRow != globalCol ? calculateEuclideanDist(
                preX, targetDim, globalRow, globalCol) : 0.0;

            double heatD = origD - diff;
            double tmpD = origD >= diff ? heatD - euclideanD : 0;
            sigma += weight * tmpD * tmpD;
        }
        return sigma;
    }

    private static double calculateEuclideanDist(
        double[][] vectors, int targetDim, int i, int j) {
        double dist = 0.0;
        for (int k = 0; k < targetDim; k++) {
            double diff = vectors[i][k] - vectors[j][k];
            dist += diff * diff;
        }

        dist = Math.sqrt(dist);
        return dist;
    }

    static double[][] generateInitMapping(int numDataPoints,
                                          int targetDim) {
        double matX[][] = new double[numDataPoints][targetDim];
        // Use Random class for generating random initial mapping solution.
        // Test the solution for the same problem by setting a constant random
        // see as shown below.
        // Random rand = new Random(47);

        // Real random seed.
        Random rand = new Random(System.currentTimeMillis());
        for (int i = 0; i < numDataPoints; i++) {
            for (int j = 0; j < targetDim; j++) {
                if(rand.nextBoolean())
                    matX[i][j] = rand.nextDouble();
                else
                    matX[i][j] = -rand.nextDouble();
            }
        }
        return matX;
    }

    private static DoubleStatistics calculateStatistics(
        RefObj<Integer> missingDistCount)
        throws MPIException {
        final DoubleStatistics[] threadDistanceSummaries =
            new DoubleStatistics[ParallelOps.threadCount];
        final int [] missingDistCounts = new int[ParallelOps.threadCount];
        IntStream.range(0, ParallelOps.threadCount).forEach(i -> missingDistCounts[i] = 0);

        if (ParallelOps.threadCount > 1) {
            launchHabaneroApp(
                () -> forallChunked(
                    0, ParallelOps.threadCount - 1,
                    (threadIdx) -> threadDistanceSummaries[threadIdx] =
                        calculateStatisticsInternal(
                            threadIdx, missingDistCounts)));
            // Sum across threads and accumulate to zeroth entry
            IntStream.range(1, ParallelOps.threadCount).forEach(
                i -> {
                    threadDistanceSummaries[0]
                        .combine(threadDistanceSummaries[i]);
                    missingDistCounts[0] += missingDistCounts[i];
                });
        }
        else {
            threadDistanceSummaries[0] = calculateStatisticsInternal(
                0, missingDistCounts);
        }

        if (ParallelOps.procCount > 1) {
            threadDistanceSummaries[0] =
                ParallelOps.allReduce(threadDistanceSummaries[0]);
            missingDistCounts[0] = ParallelOps.allReduce(missingDistCounts[0]);
        }
        missingDistCount.setValue(missingDistCounts[0]);
        return threadDistanceSummaries[0];
    }

    private static void readDistancesAndWeights() {
        distances = BinaryReader.readRowRange(
            config.distanceMatrixFile, ParallelOps.procRowRange,
            ParallelOps.globalColCount, byteOrder, config.isMemoryMapped, true);
        if (config.distanceTransform != 1.0){
            distances = BinaryReader.transform(d -> d < 0 ? d : Math.pow(d, config.distanceTransform), distances);
        }

        weights = Strings.isNullOrEmpty(config.weightMatrixFile) ? BinaryReader
            .readConstant(1.0) : BinaryReader.readRowRange(
            config.weightMatrixFile, ParallelOps.procRowRange,
            ParallelOps.globalColCount, byteOrder, config.isMemoryMapped,
            false);
    }

    private static DoubleStatistics calculateStatisticsInternal(
        int threadIdx, int[] missingDistCounts) {

        DoubleStatistics stat = new DoubleStatistics();
        int pointCount =  ParallelOps.threadRowCounts[threadIdx] *
                          ParallelOps.globalColCount;
        for (int i = 0; i < pointCount; ++i){
            int procLocalPnum =
                i + ParallelOps.threadPointStartOffsets[threadIdx];
            double origD = distances.getValue(procLocalPnum);
            double weight = weights.getValue(procLocalPnum);
            if (origD < 0) {
                // Missing distance
                ++missingDistCounts[threadIdx];
                continue;
            }
            if (weight == 0) continue; // Ignore zero weights
            stat.accept(origD);
        }
        return stat;
    }

    private static void readConfiguration(CommandLine cmd) {
        config = ConfigurationMgr.LoadConfiguration(
            cmd.getOptionValue(Constants.CMD_OPTION_LONG_C)).damdsSection;
        ParallelOps.nodeCount =
            Integer.parseInt(cmd.getOptionValue(Constants.CMD_OPTION_LONG_N));
        ParallelOps.threadCount =
            Integer.parseInt(cmd.getOptionValue(Constants.CMD_OPTION_LONG_T));
        byteOrder =
            config.isBigEndian ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
    }

    /**
     * Parse command line arguments
     *
     * @param args Command line arguments
     * @param opts Command line options
     * @return An <code>Optional&lt;CommandLine&gt;</code> object
     */
    private static Optional<CommandLine> parseCommandLineArguments(
        String[] args, Options opts) {

        CommandLineParser optParser = new GnuParser();

        try {
            return Optional.fromNullable(optParser.parse(opts, args));
        }
        catch (ParseException e) {
            System.out.println(e);
        }
        return Optional.fromNullable(null);
    }
}
