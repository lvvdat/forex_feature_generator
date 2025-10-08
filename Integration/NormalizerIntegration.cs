using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Normalization;
using ForexFeatureGenerator.Pipeline;
using Parquet;
using Parquet.Data;
using Parquet.Schema;
using System.Reflection.Metadata;

namespace ForexFeatureGenerator.Integration
{
    /// <summary>
    /// Integration example showing how to use the ComprehensiveFeatureNormalizer
    /// in both training and production scenarios
    /// </summary>
    public class NormalizerIntegration
    {
        private readonly ComprehensiveFeatureNormalizer _normalizer;
        private readonly string _normalizerPath = "normalizer_params.json";
        private StreamWriter _logWriter;

        public NormalizerIntegration(StreamWriter logWriter)
        {
            _logWriter = logWriter;
            _normalizer = new ComprehensiveFeatureNormalizer();
        }

        /// <summary>
        /// TRAINING SCENARIO: Fit normalizer on training data and save parameters
        /// This should be run once during model training
        /// </summary>
        public async Task TrainAndSaveNormalizer(string inputParquetPath, string outputParquetPath)
        {
            Log("=== TRAINING PHASE: Fitting Normalizer ===");
            Log($"Loading features from: {inputParquetPath}");

            // Step 1: Load the feature data from parquet
            var (features, labels, timestamps) = await LoadFeaturesFromParquet(inputParquetPath);

            Log($"Loaded {features.First().Value.Length} samples with {features.Count} features");

            // Step 2: Split data into train/validation sets (80/20 split)
            int totalSamples = features.First().Value.Length;
            int trainSize = (int)(totalSamples * 0.8);

            var trainFeatures = new Dictionary<string, double[]>();
            var validFeatures = new Dictionary<string, double[]>();

            foreach (var kvp in features)
            {
                trainFeatures[kvp.Key] = kvp.Value.Take(trainSize).ToArray();
                validFeatures[kvp.Key] = kvp.Value.Skip(trainSize).ToArray();
            }

            Log($"Training set: {trainSize} samples");
            Log($"Validation set: {totalSamples - trainSize} samples");

            // Step 3: Fit normalizer ONLY on training data (prevent data leakage)
            Log("\nFitting normalizer on training data...");
            _normalizer.Fit(trainFeatures);

            // Step 4: Transform both training and validation data
            Log("Transforming features...");
            var trainNormalized = _normalizer.TransformBatch(trainFeatures);
            var validNormalized = _normalizer.TransformBatch(validFeatures);

            // Step 5: Validate normalization
            Log("\nValidating normalization...");
            var stats = _normalizer.ValidateNormalization(trainNormalized);

            // Print validation summary
            var invalidFeatures = stats.Where(s => !s.Value.IsValid).ToList();
            if (invalidFeatures.Any())
            {
                Log($"Warning: {invalidFeatures.Count} features failed validation:");
                foreach (var feature in invalidFeatures.Take(5))
                {
                    var stat = feature.Value;
                    Log($"  - {stat.FeatureName}: Mean={stat.Mean:F4}, StdDev={stat.StdDev:F4}, Min={stat.Min:F4}, Max={stat.Max:F4}");
                }
            }
            else
            {
                Log("✓ All features normalized successfully");
            }

            // Check for NaN or Infinity values
            var nanFeatures = stats.Where(s => s.Value.HasNaN).Select(s => s.Key).ToList();
            var infFeatures = stats.Where(s => s.Value.HasInfinity).Select(s => s.Key).ToList();

            if (nanFeatures.Any())
                Log($"Warning: {nanFeatures.Count} features contain NaN values");
            if (infFeatures.Any())
                Log($"Warning: {infFeatures.Count} features contain Infinity values");

            // Step 6: Save normalizer parameters for production use
            Log($"\nSaving normalizer parameters to: {_normalizerPath}");
            _normalizer.SaveToFile(_normalizerPath);

            // Step 7: Save normalized data for model training
            if (!string.IsNullOrEmpty(outputParquetPath))
            {
                Log($"Saving normalized features to: {outputParquetPath}");
                await SaveNormalizedFeatures(outputParquetPath, trainNormalized, validNormalized, labels, timestamps);
            }

            // Print normalization statistics
            PrintNormalizationSummary(stats);

            Log("\n=== Training Phase Complete ===");
        }

        /// <summary>
        /// PRODUCTION SCENARIO: Load fitted normalizer and transform real-time features
        /// This is used during live trading
        /// </summary>
        public class ProductionNormalizer
        {
            private ComprehensiveFeatureNormalizer _normalizer;
            private bool _isLoaded = false;

            /// <summary>
            /// Initialize the production normalizer by loading saved parameters
            /// </summary>
            public void Initialize(string normalizerPath)
            {
                Console.WriteLine($"Loading normalizer from: {normalizerPath}");

                _normalizer = new ComprehensiveFeatureNormalizer();
                _normalizer.LoadFromFile(normalizerPath);
                _isLoaded = true;

                Console.WriteLine("✓ Production normalizer initialized");
            }

            /// <summary>
            /// Transform a single feature vector in real-time
            /// This is called for each new bar during live trading
            /// </summary>
            public FeatureVector NormalizeFeatures(FeatureVector rawFeatures)
            {
                if (!_isLoaded)
                    throw new InvalidOperationException("Normalizer not initialized. Call Initialize() first.");

                // Transform the features
                var normalizedDict = _normalizer.Transform(rawFeatures.Features);

                // Create normalized feature vector
                var normalizedFeatures = new FeatureVector
                {
                    Timestamp = rawFeatures.Timestamp,
                    Features = normalizedDict
                };

                return normalizedFeatures;
            }

            /// <summary>
            /// Batch normalization for backtesting or batch prediction
            /// </summary>
            public List<FeatureVector> NormalizeFeaturesBatch(List<FeatureVector> rawFeatures)
            {
                if (!_isLoaded)
                    throw new InvalidOperationException("Normalizer not initialized. Call Initialize() first.");

                var normalizedList = new List<FeatureVector>();

                foreach (var raw in rawFeatures)
                {
                    normalizedList.Add(NormalizeFeatures(raw));
                }

                return normalizedList;
            }

            /// <summary>
            /// Get original value from normalized value (for interpretability)
            /// </summary>
            public double GetOriginalValue(string featureName, double normalizedValue)
            {
                if (!_isLoaded)
                    throw new InvalidOperationException("Normalizer not initialized. Call Initialize() first.");

                return _normalizer.InverseTransform(featureName, normalizedValue);
            }
        }

        /// <summary>
        /// Example of integrating normalizer into the existing feature pipeline
        /// </summary>
        public class EnhancedFeaturePipeline
        {
            private readonly FeaturePipeline _basePipeline;
            private readonly ProductionNormalizer _normalizer;
            private bool _normalizationEnabled = false;

            public EnhancedFeaturePipeline(FeatureConfiguration config)
            {
                _basePipeline = new FeaturePipeline(config);
                _normalizer = new ProductionNormalizer();
            }

            /// <summary>
            /// Enable normalization by loading saved parameters
            /// </summary>
            public void EnableNormalization(string normalizerPath)
            {
                _normalizer.Initialize(normalizerPath);
                _normalizationEnabled = true;
            }

            /// <summary>
            /// Process tick and calculate normalized features
            /// </summary>
            public FeatureVector ProcessTickWithNormalization(TickData tick)
            {
                // Process tick through base pipeline
                _basePipeline.ProcessTick(tick);

                // Calculate raw features
                var rawFeatures = _basePipeline.CalculateFeatures(tick.Timestamp);

                // Apply normalization if enabled
                if (_normalizationEnabled)
                {
                    return _normalizer.NormalizeFeatures(rawFeatures);
                }

                return rawFeatures;
            }
        }

        #region Helper Methods

        /// <summary>
        /// Load features from parquet file
        /// </summary>


        private async Task<(Dictionary<string, double[]> features, int[] labels, long[] timestamps)>
            LoadFeaturesFromParquet(string path)
        {
            var features = new Dictionary<string, double[]>(StringComparer.Ordinal);
            var labelsList = new List<int>();
            var timestampsList = new List<long>();

            using (var reader = await ParquetReader.CreateAsync(path))
            {
                DataField[] dataFields = reader.Schema.GetDataFields();

                for (int i = 0; i < reader.RowGroupCount; i++)
                {
                    using (ParquetRowGroupReader rowGroup = reader.OpenRowGroupReader(i))
                    {
                        foreach (DataField field in dataFields)
                        {
                            DataColumn column = await rowGroup.ReadColumnAsync(field);

                            // NOTE: DataColumn is NOT generic. Cast its Data to the right array type.
                            if (field.Name == "label")
                            {
                                // expect Int32 labels
                                if (column.Data is int[] ints) 
                                    labelsList.AddRange(ints);
                                else 
                                    throw new InvalidOperationException($"Unexpected type for 'label': {column.Data?.GetType().Name}");
                            }
                            else if (field.Name == "timestamp")
                            {
                                // expect Int64 timestamps (e.g., epoch)
                                if (column.Data is long[] longs) 
                                    timestampsList.AddRange(longs);
                                else 
                                    throw new InvalidOperationException($"Unexpected type for 'timestamp': {column.Data?.GetType().Name}");
                            }
                            else if (field.Name.StartsWith("fg"))
                            {
                                // feature columns -> double[]
                                if (column.Data is double[] doubles)
                                {
                                    if (features.TryGetValue(field.Name, out var existing))
                                    {
                                        var combined = new double[existing.Length + doubles.Length];
                                        Array.Copy(existing, 0, combined, 0, existing.Length);
                                        Array.Copy(doubles, 0, combined, existing.Length, doubles.Length);
                                        features[field.Name] = combined;
                                    }
                                    else
                                    {
                                        // If you need to keep the dictionary independent of Parquet's buffer, clone:
                                        // features[field.Name] = (double[])doubles.Clone();
                                        features[field.Name] = doubles;
                                    }
                                }
                                else
                                {
                                    throw new InvalidOperationException($"Feature '{field.Name}' has unsupported type: {column.Data?.GetType().Name}");
                                }
                            }
                        }
                    }
                }
            }

            return (features,
                    labelsList.Count > 0 ? labelsList.ToArray() : null,
                    timestampsList.Count > 0 ? timestampsList.ToArray() : null);
        }

        /// <summary>
        /// Save normalized features to parquet
        /// </summary>
        private async Task SaveNormalizedFeatures(
                string path,
                Dictionary<string, double[]> trainFeatures,
                Dictionary<string, double[]> validFeatures,
                int[] labels,
                long[] timestamps)
        {
            // Combine train and validation data
            var allFeatures = new Dictionary<string, double[]>();

            foreach (var kvp in trainFeatures)
            {
                var trainData = kvp.Value;
                var validData = validFeatures[kvp.Key];

                var combined = new double[trainData.Length + validData.Length];
                trainData.CopyTo(combined, 0);
                validData.CopyTo(combined, trainData.Length);

                allFeatures[kvp.Key] = combined;
            }

            // Create schema
            var fields = new List<Field>();
            foreach (var featureName in allFeatures.Keys)
            {
                fields.Add(new DataField<double>(featureName));
            }

            // Add label and timestamp fields
            fields.Add(new DataField<int>("label"));
            fields.Add(new DataField<long>("timestamp"));
            fields.Add(new DataField<int>("is_train")); // Flag for train/validation split

            var schema = new ParquetSchema(fields);

            // Create is_train flag array
            var isTrainFlags = new int[labels.Length];
            int trainSize = trainFeatures.First().Value.Length;
            for (int i = 0; i < trainSize; i++)
            {
                isTrainFlags[i] = 1;
            }

            // Write to parquet
            using (var stream = System.IO.File.OpenWrite(path))
            using (var writer = await ParquetWriter.CreateAsync(schema, stream))
            {
                writer.CompressionMethod = CompressionMethod.Snappy;

                using (var rowGroup = writer.CreateRowGroup())
                {
                    // Write features
                    foreach (var kvp in allFeatures)
                    {
                        var field = schema.GetDataFields().First(f => f.Name == kvp.Key);
                        await rowGroup.WriteColumnAsync(new DataColumn(field, kvp.Value));
                    }

                    // Write labels, timestamps, and train flags
                    await rowGroup.WriteColumnAsync(new DataColumn(schema.GetDataFields().First(f => f.Name == "label"), labels));
                    await rowGroup.WriteColumnAsync(new DataColumn(schema.GetDataFields().First(f => f.Name == "timestamp"), timestamps));
                    await rowGroup.WriteColumnAsync(new DataColumn(schema.GetDataFields().First(f => f.Name == "is_train"), isTrainFlags));
                }
            }
        }

        /// <summary>
        /// Print summary of normalization statistics
        /// </summary>
        private void PrintNormalizationSummary(
            Dictionary<string, ComprehensiveFeatureNormalizer.NormalizationStats> stats)
        {
            Log("\n=== Normalization Summary ===");

            // Group by scaler type
            var grouped = stats.GroupBy(s => s.Value.ScalerType);

            foreach (var group in grouped)
            {
                var scalerType = group.Key;
                var features = group.ToList();

                Log($"\n{scalerType}: {features.Count} features");

                // Calculate aggregate statistics
                var validCount = features.Count(f => f.Value.IsValid);
                var avgMean = features.Average(f => Math.Abs(f.Value.Mean));
                var avgStd = features.Average(f => f.Value.StdDev);

                Log($"  Valid: {validCount}/{features.Count}");
                Log($"  Avg |Mean|: {avgMean:F4}");
                Log($"  Avg StdDev: {avgStd:F4}");

                // Show sample features
                Log($"  Sample features: {string.Join(", ", features.Take(3).Select(f => f.Key))}");
            }

            // Overall statistics
            Log($"\nTotal features: {stats.Count}");
            Log($"Features normalized: {stats.Count(s => s.Value.ScalerType != "None")}");
            Log($"Features unchanged: {stats.Count(s => s.Value.ScalerType == "None")}");
        }

        private void Log(string message, ConsoleColor color = ConsoleColor.White)
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss");
            var logMessage = $"[{timestamp}] {message}";

            Console.ForegroundColor = color;
            Console.WriteLine(message);
            Console.ResetColor();

            _logWriter?.WriteLine(logMessage);
        }

        #endregion
    }
}