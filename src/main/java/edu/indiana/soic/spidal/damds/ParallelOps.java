package edu.indiana.soic.spidal.damds;

import edu.indiana.soic.spidal.common.DoubleStatistics;
import edu.indiana.soic.spidal.common.Range;
import edu.indiana.soic.spidal.common.RangePartitioner;
import mpi.Intracomm;
import mpi.MPI;
import mpi.MPIException;

import java.nio.ByteBuffer;
import java.util.stream.IntStream;

public class ParallelOps {
    public static int nodeCount=1;
    public static int threadCount=1;

    public static Intracomm procComm;
    public static int procRank;
    public static int procCount;
    public static String parallelPattern;

    public static Range procRowRange;
    public static int procRowStartOffset;
    public static int procRowCount;
    public static int procPointStartOffset;

    public static Range[] threadRowRanges;
    public static int[] threadRowStartOffsets;
    public static int[] threadRowCounts;
    public static int[] threadPointStartOffsets;


    public static int globalColCount;

    // Buffers for MPI operations
    private static ByteBuffer statBuffer;

    public static void setupParallelism(String[] args) throws MPIException {
        MPI.Init(args);
        procComm = MPI.COMM_WORLD; //initializing MPI world communicator
        procRank = procComm.getRank();
        procCount = procComm.getSize();

        int mpiPerNode = procCount / nodeCount;

        if ((mpiPerNode * nodeCount) != procCount) {
            Utils.printAndThrowRuntimeException(
                "Inconsistent MPI counts Nodes " + nodeCount + " Size " +
                procCount);
        }

        statBuffer = MPI.newByteBuffer(DoubleStatistics.extent);

        parallelPattern =
            "---------------------------------------------------------\n" +
            "Machine:" + MPI.getProcessorName() + " " +
            threadCount + "x" + mpiPerNode + "x" + nodeCount;
        Utils.printMessage(parallelPattern);
    }

    public static void tearDownParallelism() throws MPIException {
        // End MPI
        MPI.Finalize();
    }

    public static void setParallelDecomposition(int globalRowCount) {
        //	First divide points among processes
        Range[] rowRanges = RangePartitioner.partition(globalRowCount,
                                                       procCount);
        Range rowRange = rowRanges[procRank]; // The range of points for this process

        procRowRange = rowRange;
        procRowStartOffset = rowRange.getStartIndex();
        procRowCount = rowRange.getLength();
        globalColCount = globalRowCount;
        procPointStartOffset = procRowStartOffset * globalColCount;

        // Next partition points per process among threads
        threadRowRanges = RangePartitioner.partition(procRowCount, threadCount);
        threadRowCounts = new int[threadCount];
        threadRowStartOffsets = new int[threadCount];
        threadPointStartOffsets = new int[threadCount];
        IntStream.range(0, threadCount).parallel().forEach(
            threadIdx -> {
                Range threadRowRange = threadRowRanges[threadIdx];
                threadRowCounts[threadIdx] = threadRowRange.getLength();
                threadRowStartOffsets[threadIdx] =
                    threadRowRange.getStartIndex();
                threadPointStartOffsets[threadIdx] =
                    threadRowStartOffsets[threadIdx] * globalColCount;
            });
    }

    public static DoubleStatistics allReduce(DoubleStatistics stat) throws
        MPIException {
        stat.addToBuffer(statBuffer,0);
        procComm.allReduce(
            statBuffer, DoubleStatistics.extent, MPI.BYTE,
            DoubleStatistics.reduceSummaries());
        return DoubleStatistics.getFromBuffer(statBuffer, 0);
    }
}
