using System.Linq;
using System.Text;
using System.Diagnostics;

using Parquet;
using Parquet.Schema;
using Parquet.Data;

using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Advanced;
using ForexFeatureGenerator.Features.M1;
using ForexFeatureGenerator.Features.M5;
using ForexFeatureGenerator.Label;
using ForexFeatureGenerator.Pipeline;
using ForexFeatureGenerator.Utilities;

namespace ForexFeatureGenerator
{
    class Program
    {
        // ============= CONFIGURATION =============
        private static readonly string OUTPUT_DIR = "processed";
        private static readonly string LOG_FILE = Path.Combine("logs", $"log_{DateTime.Now:yyyyMMdd_HHmmss}.log");

        // Label Generation Configuration
        private static readonly LabelGenerationConfig LABEL_CONFIG = new()
        {
            TriggerPips = 3.5,
            DistancePips = 2.5,
            MaxFutureTicks = 600,
            MinConfidenceThreshold = 0.3,
            MinScoreThreshold = 0.35
        };

        private static StreamWriter _logWriter;
        private static Stopwatch _globalStopwatch = new Stopwatch();

        static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;

            _globalStopwatch.Start();

            // Setup
            if (!Directory.Exists(OUTPUT_DIR))
                Directory.CreateDirectory(OUTPUT_DIR);

            if (!Directory.Exists("logs"))
                Directory.CreateDirectory("logs");
            _logWriter = new StreamWriter(LOG_FILE, false, Encoding.UTF8) { AutoFlush = true };

            // ===== PHASE 1: DATA LOADING =====
            Log("PHASE 1: DATA LOADING", ConsoleColor.Cyan);
            Log("━".PadRight(60, '━'), ConsoleColor.Cyan);

            var tickData = await LoadTickDataAsync("data/ticks_data.csv");
            ValidateTickData(tickData);

            // ===== PHASE 2: LABEL GENERATION =====
            Log("\nPHASE 2: LABEL GENERATION", ConsoleColor.Cyan);
            Log("━".PadRight(60, '━'), ConsoleColor.Cyan);

            var outputPath = Path.Combine(OUTPUT_DIR, $"features_labels_{DateTime.Now:yyyyMMdd_HHmmss}.parquet");
            var generatedLabels = await GenerateFeaturesAndLabelsAsync(tickData, outputPath);

            AnalyzeGeneratedLabels(generatedLabels);
        }


        // ============= DATA LOADING FUNCTIONS =============
        static async Task<List<TickData>> LoadTickDataAsync(string path)
        {
            return await Task.Run(async () =>
            {
                Log("Loading tick data...");

                if (!File.Exists(path))
                {
                    throw new FileNotFoundException($"  ⚠️ Tick data file not found: {path}");
                }

                var ticks = await TickLoader.LoadTickDataAsync(path);

                Log($"  ✓ Loaded {ticks.Count:N0} ticks");

                return ticks;
            });
        }

        static void ValidateTickData(List<TickData> ticks)
        {
            Log("\nValidating tick data quality...");

            var issues = new List<string>();

            // Check spread statistics
            var spreads = ticks.Select(t => t.Spread).ToList();
            var avgSpread = spreads.Average();
            var maxSpread = spreads.Max();
            var minSpread = spreads.Min();

            Log($"  Spread Statistics:");
            Log($"    Average: {avgSpread * 10000:F2} pips");
            Log($"    Min: {minSpread * 10000:F2} pips");
            Log($"    Max: {maxSpread * 10000:F2} pips");

            if (maxSpread > avgSpread * 10)
            {
                issues.Add($"Extreme spread detected: {maxSpread * 10000:F2} pips");
            }

            // Check for time gaps
            for (int i = 1; i < Math.Min(ticks.Count, 1000); i++)
            {
                if (ticks[i].Timestamp < ticks[i - 1].Timestamp)
                {
                    issues.Add($"Time ordering issue at index {i}");
                }
            }

            // Check data density
            if (ticks.Count > 1)
            {
                var totalTime = ticks.Last().Timestamp - ticks.First().Timestamp;
                var avgTicksPerMinute = ticks.Count / Math.Max(totalTime.TotalMinutes, 1);
                Log($"  Data Density: {avgTicksPerMinute:F1} ticks/minute");

                if (avgTicksPerMinute < 1)
                {
                    issues.Add("Low data density (< 1 tick/minute)");
                }
            }

            if (issues.Count > 0)
            {
                Log($"\n  ⚠️ Data quality issues:", ConsoleColor.Yellow);
                foreach (var issue in issues.Take(5))
                {
                    Log($"    - {issue}", ConsoleColor.Yellow);
                }
            }
            else
            {
                Log("  ✓ Data validation passed");
            }
        }

        // ============= LABEL GENERATION FUNCTIONS =============
        static async Task<List<LabelResult>>
            GenerateFeaturesAndLabelsAsync(List<TickData> tickData, string outputPath)
        {
            return await Task.Run(async () =>
            {
                Log("Building feature pipeline...");
                var pipeline = BuildComprehensivePipeline();

                var labelResults = new List<LabelResult>();
                var m1Aggregator = pipeline.GetAggregator(TimeSpan.FromMinutes(1));

                int barsProcessed = 0;
                int warmupBars = 275;
                int futureTicksNeeded = LABEL_CONFIG.MaxFutureTicks;

                var maxTicks = tickData.Count;
                var progress = new ProgressReporter("Generating features and labels", maxTicks);

                // Parquet-related (lazy init)
                ParquetWriter? parquetWriter = null;
                FileStream? outStream = null;
                ParquetSchema? parquetSchema = null;

                List<string>? featureNames = null;

                // Cached fields (avoid schema lookups later)
                List<DataField<double>>? featureFields = null;
                DataField<int>? labelField = null;
                DataField<double>? confidenceField = null;
                DataField<double>? longPipsField = null;
                DataField<double>? shortPipsField = null;

                const int BatchSize = 100_000;
                Dictionary<string, List<double>>? featureBuffers = null;
                List<int>? labelBuffer = null;
                List<double>? confidenceBuffer = null;
                List<double>? longPipsBuffer = null;
                List<double>? shortPipsBuffer = null;

                async Task FlushBatchAsync()
                {
                    if (parquetWriter is null || parquetSchema is null || featureBuffers is null ||
                        labelBuffer is null || labelBuffer.Count == 0) return;

                    using (var rowGroup = parquetWriter.CreateRowGroup())
                    {
                        // features (preserve order)
                        for (int idx = 0; idx < featureNames!.Count; idx++)
                        {
                            var fname = featureNames[idx];
                            var f = featureFields![idx];
                            await rowGroup.WriteColumnAsync(new DataColumn(f, featureBuffers[fname].ToArray()));
                        }

                        await rowGroup.WriteColumnAsync(new DataColumn(labelField!, labelBuffer.ToArray()));
                        await rowGroup.WriteColumnAsync(new DataColumn(confidenceField!, confidenceBuffer!.ToArray()));
                        await rowGroup.WriteColumnAsync(new DataColumn(longPipsField!, longPipsBuffer!.ToArray()));
                        await rowGroup.WriteColumnAsync(new DataColumn(shortPipsField!, shortPipsBuffer!.ToArray()));
                    }

                    foreach (var kv in featureBuffers!) kv.Value.Clear();
                    labelBuffer!.Clear(); confidenceBuffer!.Clear(); longPipsBuffer!.Clear(); shortPipsBuffer!.Clear();
                }

                if (File.Exists(outputPath)) File.Delete(outputPath);

                try
                {
                    for (int i = 0; i < maxTicks; i++)
                    {
                        pipeline.ProcessTick(tickData[i]);

                        var completedBar = m1Aggregator?.GetCompletedBar();
                        if (completedBar != null)
                        {
                            barsProcessed++;

                            var features = pipeline.CalculateFeatures(completedBar.Timestamp);

                            var currentTick = tickData[i];
                            var futureTicks = tickData.Skip(i + 1).Take(futureTicksNeeded).ToList();

                            var labelResult = LabelGenerator.GenerateLabel(LABEL_CONFIG, currentTick, futureTicks);

                            if (labelResult != null)
                            {
                                labelResults.Add(labelResult);

                                if (barsProcessed > warmupBars)
                                {
                                    // lazy init once we know feature names
                                    if (parquetWriter is null)
                                    {
                                        featureNames = features.Features.Keys.ToList();

                                        var fields = new List<Field>(featureNames.Count + 4);

                                        featureFields = new List<DataField<double>>(featureNames.Count);
                                        foreach (var fname in featureNames)
                                        {
                                            var f = new DataField<double>(fname);
                                            featureFields.Add(f);
                                            fields.Add(f);
                                        }

                                        labelField = new DataField<int>("label");
                                        confidenceField = new DataField<double>("confidence");
                                        longPipsField = new DataField<double>("long_profit_pips");
                                        shortPipsField = new DataField<double>("short_profit_pips");

                                        fields.Add(labelField);
                                        fields.Add(confidenceField);
                                        fields.Add(longPipsField);
                                        fields.Add(shortPipsField);

                                        parquetSchema = new ParquetSchema(fields);

                                        outStream = File.Create(outputPath);
                                        parquetWriter = await ParquetWriter.CreateAsync(parquetSchema, outStream);
                                        parquetWriter.CompressionMethod = CompressionMethod.Snappy;

                                        featureBuffers = featureNames.ToDictionary(n => n, _ => new List<double>(BatchSize));
                                        labelBuffer = new List<int>(BatchSize);
                                        confidenceBuffer = new List<double>(BatchSize);
                                        longPipsBuffer = new List<double>(BatchSize);
                                        shortPipsBuffer = new List<double>(BatchSize);
                                    }

                                    try
                                    {
                                        // append row into buffers
                                        foreach (var fname in featureNames!)
                                        {
                                            if (features.Features.TryGetValue(fname, out var val))
                                                featureBuffers![fname].Add(val);
                                            else
                                                featureBuffers![fname].Add(0.0);
                                        }

                                        labelBuffer!.Add(labelResult.Label);
                                        confidenceBuffer!.Add(labelResult.Confidence);
                                        longPipsBuffer!.Add(labelResult.LongProfitPips);
                                        shortPipsBuffer!.Add(labelResult.ShortProfitPips);

                                        if (labelBuffer!.Count >= BatchSize)
                                            await FlushBatchAsync();
                                    }
                                    catch (Exception ex)
                                    {
                                        Log($"  ⚠️ Error buffering data at bar {barsProcessed}: {ex.Message}", ConsoleColor.Yellow);
                                    }
                                }
                            }
                        }

                        if (i % 1000 == 0)
                            progress.Update(i, $"Bars: {barsProcessed}, Labels: {labelResults.Count}");
                    }

                    await FlushBatchAsync();

                    progress.Complete();
                    Log($"  Processed {barsProcessed} M1 bars");
                    Log($"  Generated {labelResults.Count} feature vectors with labels");

                    return labelResults;
                }
                finally
                {
                    if (parquetWriter is not null) await parquetWriter.DisposeAsync();
                    outStream?.Dispose();
                }
            });
        }


        static FeaturePipeline BuildComprehensivePipeline()
        {
            var config = FeatureConfiguration.CreateDefault();
            var pipeline = new FeaturePipeline(config);

            // Register multiple timeframes
            var timeframes = new[]
            {
                (TimeSpan.FromMinutes(1), 1000),
                (TimeSpan.FromMinutes(5), 500),
            };

            foreach (var (tf, size) in timeframes)
            {
                pipeline.RegisterAggregator(tf, size);
            }

            RegisterAllFeatureCalculators(pipeline);

            return pipeline;
        }

        static void RegisterAllFeatureCalculators(FeaturePipeline pipeline)
        {
            // M1 Features
            pipeline.RegisterCalculator(new M1ComprehensiveFeatures());
            pipeline.RegisterCalculator(new M1MicrostructureFeatures());
            pipeline.RegisterCalculator(new M1MomentumFeatures());
            pipeline.RegisterCalculator(new M1VolatilityFeatures());

            // Advanced Features
            pipeline.RegisterCalculator(new OrderFlowFeatures());
            pipeline.RegisterCalculator(new MarketRegimeFeatures());
            pipeline.RegisterCalculator(new PatternRecognitionFeatures());
            pipeline.RegisterCalculator(new LiquidityFeatures());

            // M5 Features
            pipeline.RegisterCalculator(new M5TrendFeatures());
            pipeline.RegisterCalculator(new M5MomentumFeatures());
            pipeline.RegisterCalculator(new M5VolatilityFeatures());
            pipeline.RegisterCalculator(new M5VolumeFeatures());
            pipeline.RegisterCalculator(new M5OscillatorFeatures());

            Log($"  ✓ Registered {14} features");
        }

        static void AnalyzeGeneratedLabels(List<LabelResult> labels)
        {
            Log("\nAnalyzing generated labels...");

            var distribution = labels.GroupBy(l => l.Label)
                .ToDictionary(g => g.Key, g => g.Count());

            var total = labels.Count;
            Log("  Label Distribution:");
            Log($"    LONG (1):     {distribution.GetValueOrDefault(1, 0),10} ({(double)distribution.GetValueOrDefault(1, 0) / total:P2})");
            Log($"    SHORT (-1):   {distribution.GetValueOrDefault(-1, 0),10} ({(double)distribution.GetValueOrDefault(-1, 0) / total:P2})");
            Log($"    NEUTRAL (0):  {distribution.GetValueOrDefault(0, 0),10} ({(double)distribution.GetValueOrDefault(0, 0) / total:P2})");

            var confidences = labels.Select(l => l.Confidence).ToList();
            Log("\n  Confidence Statistics:");
            Log($"    Average: {confidences.Average():F3}");
            Log($"    Min: {confidences.Min():F3}");
            Log($"    Max: {confidences.Max():F3}");
        }

        static void Log(string message, ConsoleColor color = ConsoleColor.White)
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss");
            var logMessage = $"[{timestamp}] {message}";

            Console.ForegroundColor = color;
            Console.WriteLine(message);
            Console.ResetColor();

            _logWriter?.WriteLine(logMessage);
        }
    }
}