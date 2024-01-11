using Hybridizer.Runtime.CUDAImports;
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace Cuda_Test
{
    class CudaTest
    {

        public static readonly int MAX_PARALLEL_CORE = 16;
        
        // Set Count as high as possible. Remember higher the number longer it will take to run
        public static readonly int TEST_COUNT_NUMBER = 68000000;
        private static double[] a = new double[TEST_COUNT_NUMBER], b = new double[TEST_COUNT_NUMBER], c = new double[TEST_COUNT_NUMBER];

        static void Main(string[] args)
        {
            Console.WriteLine("Configuration ...");

            SetupData();

            Console.WriteLine("Use GPU ? (y , n )");
            if (Console.ReadKey().Equals('y'))
                RunOnGPU(a, b);
            else
                RunOnCPU(a, b);

            Console.ReadKey(); Console.ReadKey();
        }

        internal static void SetupData()
        {
            int MsgFreq = TEST_COUNT_NUMBER / 5;
            Random generator = new Random();
            Parallel.For(
                0,
                TEST_COUNT_NUMBER - 1,
                new ParallelOptions { MaxDegreeOfParallelism = MAX_PARALLEL_CORE },
                i =>
                {
                    a[i] = generator.Next(int.MaxValue);
                    Thread.Sleep((int)a[i] % 3);
                    b[i] = generator.Next(int.MaxValue);
                    if ((i + 1) % MsgFreq == 0)
                        Console.WriteLine("\tIteration " + (i + 1));
                });
        }

        [EntryPoint]
        internal static void RunOnGPU(double[] a, double[] b)
        {
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            //if .SetDistrib is not used, the default is .SetDistrib(prop.multiProcessorCount * 16, 128)
            HybRunner runner = HybRunner.Cuda();
            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new LoadTest());

            Console.Out.WriteLine("\n[" + DateTime.Now.ToLongTimeString() + "] Started heavy load task on GPU.");
            var executionWatch = Stopwatch.StartNew();

            wrapped.Run(a, b);

            executionWatch.Stop();
            Console.Out.WriteLine("[" + DateTime.Now.ToLongTimeString() + "] Finished heavy load task.");
            double elapsedS = Math.Floor((double)executionWatch.ElapsedMilliseconds / 1000), elapsedMs = executionWatch.ElapsedMilliseconds % 1000;
            Console.Out.WriteLine("The time elapsed during the load is " + elapsedS + " s " + elapsedMs + " ms.");
        }

        public static void Run(double[] a, double[] b)
        {
            Parallel.For(
                0,
                TEST_COUNT_NUMBER - 1,
                new ParallelOptions { MaxDegreeOfParallelism = MAX_PARALLEL_CORE },
                i => {
                    double auxA = a[i];
                    a[i] = Math.Atan(Math.Log10(a[i] / 13021) + Math.Log10(Math.Min(int.MaxValue, Math.Pow((b[i] / 231232423), 4))));
                    b[i] = Math.Atan2(Math.Log10(auxA), Math.Log10(b[i]));
                    c[i] = Math.Log10(Math.Sqrt(Math.Pow(a[i] - b[i], 2)));
                });
        }

        internal static void RunOnCPU(double[] a, double[] b)
        {
            Console.WriteLine("[" + DateTime.Now.ToLongTimeString() + "] Started heavy load task on CPU.");
            var executionWatch = Stopwatch.StartNew();

            Run(a, b);

            executionWatch.Stop();
            Console.WriteLine("[" + DateTime.Now.ToLongTimeString() + "] Compiled Task");
            double elapsedS = Math.Floor((double)executionWatch.ElapsedMilliseconds / 1000), elapsedMs = executionWatch.ElapsedMilliseconds % 1000;
            Console.WriteLine("The time elapsed during the load is " + elapsedS + " s " + elapsedMs + " ms.");
        }
    }
