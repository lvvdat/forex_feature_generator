using System.Text;
using System.Diagnostics;

using Parquet;
using Parquet.Data;
using Parquet.Schema;

using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Label;
using ForexFeatureGenerator.Pipeline;
using ForexFeatureGenerator.Utilities;
using ForexFeatureGenerator.Features.Pipeline;

namespace ForexFeatureGenerator
{
    class Program
    {
        // ============= CONFIGURATION =============
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

        private static StreamWriter _logWriter = new StreamWriter(LOG_FILE, false, Encoding.UTF8) { AutoFlush = true };

        static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;

            var inputPath = args.Length > 0 ? args[0] : "data/ticks_data.csv";
            var outputDir = args.Length > 1 ? args[1] : "output";

            // Setup
            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);

            if (!Directory.Exists("logs"))
                Directory.CreateDirectory("logs");

            var outputPath = Path.Combine(outputDir, "features_labels.parquet");

            if (!File.Exists(outputPath))
            {
                // ===== PHASE 1: DATA LOADING =====
                Log("PHASE 1: DATA LOADING", ConsoleColor.Cyan);
                Log("━".PadRight(60, '━'), ConsoleColor.Cyan);

                var tickData = await LoadTickDataAsync(inputPath);
                ValidateTickData(tickData);

                // ===== PHASE 2: LABEL GENERATION =====
                Log("\nPHASE 2: LABEL GENERATION", ConsoleColor.Cyan);
                Log("━".PadRight(60, '━'), ConsoleColor.Cyan);

                await GenerateFeaturesAndLabelsAsync(tickData, outputPath);                
            }
            else
            {
                Log($"Features and labels file already exists at: {outputPath}", ConsoleColor.Yellow);
            }
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

            // Check data density
            if (ticks.Count > 1)
            {
                for (int i = 1; i < ticks.Count; i++)
                {
                    if (ticks[i].Timestamp < ticks[i - 1].Timestamp)
                    {
                        issues.Add($"Time ordering issue at index {i}");
                    }
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
        static async Task GenerateFeaturesAndLabelsAsync(List<TickData> tickData, string outputPath)
        {
            await Task.Run(async () =>
            {
                Log("Building feature pipeline...");
                var pipeline = BuildComprehensivePipeline();

                var labelResults = new List<LabelResult>();
                var m1Aggregator = pipeline.GetAggregator(TimeSpan.FromMinutes(1));

                int barsProcessed = 0;
                int warmupBars = 255;
                int futureTicksNeeded = LABEL_CONFIG.MaxFutureTicks;

                var maxTicks = tickData.Count;
                var progress = new ProgressReporter("Generating features and labels", maxTicks);

                // Parquet-related (lazy init)
                ParquetWriter? parquetWriter = null;
                FileStream? outStream = null;
                ParquetSchema? parquetSchema = null;

                List<string>? featureNames = null;

                const int BatchSize = 100_000;

                // Cached fields (avoid schema lookups later)

                List<DataField<double>>? featureFields = null;
                Dictionary<string, List<double>>? featureBuffers = null;

                List<int>? labelBuffer = null;
                DataField<int>? labelField = null;

                List<long>? timeBuffer = null;
                DataField<long>? timeField = null;

                async Task FlushBatchAsync()
                {
                    if (parquetWriter is null || parquetSchema is null || featureBuffers is null || labelBuffer is null || labelBuffer.Count == 0) return;

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
                        await rowGroup.WriteColumnAsync(new DataColumn(timeField!, timeBuffer!.ToArray()));
                    }

                    foreach (var kv in featureBuffers!)
                        kv.Value.Clear();

                    labelBuffer!.Clear();
                    timeBuffer!.Clear();
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

                            if (labelResult != null && features != null)
                            {
                                if (barsProcessed > warmupBars)
                                {
                                    if (features.Features.Count != 146)
                                    {
                                        Log($"  ⚠️ Not enough features generated at bar {barsProcessed} ({features.Features.Count})", ConsoleColor.Yellow);
                                    }

                                    // lazy init once we know feature names
                                    if (parquetWriter is null)
                                    {
                                        featureNames = features.Features.Keys.ToList();

                                        var fields = new List<Field>(featureNames.Count + 5);

                                        featureFields = new List<DataField<double>>(featureNames.Count);
                                        foreach (var fname in featureNames)
                                        {
                                            var f = new DataField<double>(fname);
                                            featureFields.Add(f);
                                            fields.Add(f);
                                        }

                                        labelField = new DataField<int>("label");
                                        fields.Add(labelField);

                                        timeField = new DataField<long>("timestamp");
                                        fields.Add(timeField);

                                        parquetSchema = new ParquetSchema(fields);

                                        outStream = File.Create(outputPath);
                                        parquetWriter = await ParquetWriter.CreateAsync(parquetSchema, outStream);
                                        parquetWriter.CompressionMethod = CompressionMethod.Snappy;

                                        featureBuffers = featureNames.ToDictionary(n => n, _ => new List<double>(BatchSize));
                                        
                                        labelBuffer = new List<int>(BatchSize);
                                        timeBuffer = new List<long>(BatchSize);
                                    }

                                    try
                                    {
                                        labelResults.Add(labelResult);

                                        // append row into buffers
                                        foreach (var fname in featureNames!)
                                        {
                                            if (features.Features.TryGetValue(fname, out var val))
                                                featureBuffers![fname].Add(val);
                                            else
                                                featureBuffers![fname].Add(0.0);
                                        }

                                        labelBuffer!.Add(labelResult.Label);
                                        timeBuffer!.Add(completedBar.Timestamp.Ticks);

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

                    AnalyzeGeneratedLabels(labelResults);
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
            var config = FeatureConfiguration.CreateOptimized3Class();
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

            return pipeline;
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