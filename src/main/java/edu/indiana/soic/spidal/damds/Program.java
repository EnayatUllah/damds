package edu.indiana.soic.spidal.damds;

import com.google.common.base.Optional;
import edu.indiana.soic.spidal.common.BinaryReader;
import edu.indiana.soic.spidal.common.DoubleStatistics;
import edu.indiana.soic.spidal.common.Range;
import edu.indiana.soic.spidal.configuration.ConfigurationMgr;
import edu.indiana.soic.spidal.configuration.section.DAMDSSection;
import edu.rice.hj.api.SuspendableException;
import mpi.MPIException;
import org.apache.commons.cli.*;

import java.nio.ByteOrder;
import java.util.Arrays;
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

        System.out.println("== DAMDS run started on " + new Date() + " ==");

        //  Read Metadata using this as source of other metadata
        readConfiguration(cmd);
        System.out.println(config.toString(true));

        try {
            //  Set up MPI and threads parallelism
            ParallelOps.setupParallelism(args);
            ParallelOps.setParallelDecomposition(config.numberDataPoints);

            readDistancesAndWeights();
            DoubleStatistics distanceSummary = computeStatistics();
            Utils.printMessage(distanceSummary.toString());

            double[][] preX = generateInitMapping(
                config.numberDataPoints, config.targetDimension);
            double tCur = 0.0;
            double preStress = calculateStress(preX, tCur);


            ParallelOps.tearDownParallelism();
        }
        catch (MPIException e) {
            Utils.printAndThrowRuntimeException(new RuntimeException(e));
        }
    }

    private static double calculateStress(double[][] preX, double tCur) {


    }

    static double[][] generateInitMapping(int numDataPoints,
                                          int targetDim) {
        double matX[][] = new double[numDataPoints][targetDim];
        // Use Random class for generating random initial mapping solution.
        // Test the solution for the same problem by setting a constant random
        // see as shown below.
        // Random rand = new Random(47);
        Random rand = new Random(System.currentTimeMillis()); // Real random
        // seed.
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

    private static DoubleStatistics computeStatistics()
        throws MPIException {
        final DoubleStatistics[] threadDistanceSummaries =
            new DoubleStatistics[ParallelOps.threadCount];

        if (ParallelOps.threadCount > 1) {
            launchHabaneroApp(
                () -> forallChunked(
                    0, ParallelOps.threadCount - 1,
                    (threadIdx) -> threadDistanceSummaries[threadIdx] =
                        summarizeDistances(threadIdx, config.isSammon)));
            // Sum across threads and accumulate to zeroth entry
            IntStream.range(1, ParallelOps.threadCount).forEach(
                threadIdx -> threadDistanceSummaries[0]
                    .combine(threadDistanceSummaries[threadIdx]));
        }
        else {
            threadDistanceSummaries[0] =
                summarizeDistances(0, config.isSammon);
        }

        if (ParallelOps.procCount > 1) {
            threadDistanceSummaries[0] =
                ParallelOps.allReduce(threadDistanceSummaries[0]);
        }
        return threadDistanceSummaries[0];
    }

    private static void readDistancesAndWeights() {
        distances = BinaryReader.readRowRange(
            config.distanceMatrixFile, ParallelOps.procRowRange,
            ParallelOps.globalColCount, byteOrder, config.isMemoryMapped, true);

        if (!config.isSammon){
            weights = BinaryReader.readRowRange(
                config.weightMatrixFile, ParallelOps.procRowRange,
                ParallelOps.globalColCount, byteOrder,
                config.isMemoryMapped, false);
        }
    }

    private static DoubleStatistics summarizeDistances(
        int threadIdx, boolean isSammon) {
        if (isSammon) {
            // Sammon mode
            // Use all distances
            return IntStream.range(
                0, ParallelOps.threadRowCounts[threadIdx] *
                   ParallelOps.globalColCount).mapToDouble(
                i -> {
                    int procLocalPnum =
                        i + ParallelOps.threadPointStartOffsets[threadIdx];
                    double d = distances.getValue(procLocalPnum);
                    return config.distanceTransform != 1.0 ? Math.pow(d, config.distanceTransform) : d;
                }).collect(
                DoubleStatistics::new, DoubleStatistics::accept,
                DoubleStatistics::combine);
        }
        else {
            // Non Sammon mode
            // Use only distances that have non zero corresponding weights
            return IntStream.range(
                0, ParallelOps.threadRowCounts[threadIdx] *
                   ParallelOps.globalColCount).filter(
                i -> {
                    int procLocalPnum =
                        i + ParallelOps.threadPointStartOffsets[threadIdx];
                    return weights.getValue(procLocalPnum) != 0;
                }).mapToDouble(
                i -> {
                    int procLocalPnum =
                        i + ParallelOps.threadPointStartOffsets[threadIdx];
                    double d = distances.getValue(procLocalPnum);
                    return config.distanceTransform != 1.0 ? Math.pow(d, config.distanceTransform) : d;
                }).collect(
                DoubleStatistics::new, DoubleStatistics::accept,
                DoubleStatistics::combine);
        }
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
