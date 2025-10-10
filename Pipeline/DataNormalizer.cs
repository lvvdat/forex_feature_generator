using Parquet;
using Parquet.Data;
using Parquet.Schema;

using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Core.Statistics;
using ForexFeatureGenerator.Utilities;

namespace ForexFeatureGenerator.Pipeline
{
    /// <summary>
    /// Normalizes feature data using collected statistics
    /// Processes data in chunks to handle large files with limited RAM
    /// </summary>
    public class DataNormalizer
    {
        private readonly NormalizationConfig _config;
        private readonly FeatureStatisticsCollector _statistics;
        private readonly int _chunkSize;
        private readonly StreamWriter _logWriter;

        public DataNormalizer(FeatureStatisticsCollector statistics, StreamWriter logWriter, int chunkSize = 100000)
        {
            _chunkSize = chunkSize;
            _statistics = statistics;
            _logWriter = logWriter;
            _config = new NormalizationConfig();
        }

        public async Task NormalizeDataAsync(string inputPath, string outputPath, string? statsOutputPath = null)
        {
            Log($"Starting data normalization...");
            Log($"  Input: {inputPath}");
            Log($"  Output: {outputPath}");
            Log($"  Chunk size: {_chunkSize:N0} rows");

            // Save statistics if path provided
            if (!string.IsNullOrEmpty(statsOutputPath))
            {
                _statistics.SaveStatistics(statsOutputPath);
                Log($"  Statistics saved to: {statsOutputPath}");
            }

            // Get feature statistics
            var featureStats = _statistics.GetStatistics();

            // Process file in chunks
            using var inputStream = File.OpenRead(inputPath);
            using var reader = await ParquetReader.CreateAsync(inputStream);

            var schema = reader.Schema;
            var rowGroupCount = reader.RowGroupCount;

            Log($"  Input file has {rowGroupCount} row groups");

            // Create output file
            using var outputStream = File.Create(outputPath);
            using var writer = await ParquetWriter.CreateAsync(schema, outputStream);
            writer.CompressionMethod = CompressionMethod.Snappy;

            long totalRowsProcessed = 0;
            var progress = new ProgressReporter("Normalizing", (int)reader.Metadata!.NumRows);

            // Process each row group
            for (int rgIndex = 0; rgIndex < rowGroupCount; rgIndex++)
            {
                using var rowGroupReader = reader.OpenRowGroupReader(rgIndex);
                var groupSize = (int)rowGroupReader.RowCount;

                // Read all columns for this row group
                var columns = new Dictionary<string, Array>();
                var dataFields = schema.DataFields;

                foreach (var field in dataFields)
                {
                    var column = await rowGroupReader.ReadColumnAsync(field);
                    columns[field.Name] = column.Data;
                }

                // Process in smaller chunks within the row group
                for (int chunkStart = 0; chunkStart < groupSize; chunkStart += _chunkSize)
                {
                    int chunkEnd = Math.Min(chunkStart + _chunkSize, groupSize);
                    int chunkRows = chunkEnd - chunkStart;

                    // Create normalized data for this chunk
                    var normalizedColumns = new Dictionary<string, Array>();

                    foreach (var field in dataFields)
                    {
                        var columnData = columns[field.Name];

                        // Check if this is a feature column (not label or timestamp)
                        if (field.Name == "label")
                        {
                            // Copy metadata columns as-is
                            normalizedColumns[field.Name] = CopyIntArraySlice(columnData, chunkStart, chunkRows);
                        }
                        else if (field.Name == "timestamp")
                        {
                            normalizedColumns[field.Name] = CopyLongArraySlice(columnData, chunkStart, chunkRows);
                        }
                        else
                        {
                            // Normalize feature column
                            var normalizedData = NormalizeColumn(
                                field.Name,
                                columnData,
                                chunkStart,
                                chunkRows,
                                featureStats.GetValueOrDefault(field.Name));

                            normalizedColumns[field.Name] = normalizedData;
                        }
                    }

                    // Write normalized chunk to output
                    await WriteChunkAsync(writer, schema, normalizedColumns, chunkRows);

                    totalRowsProcessed += chunkRows;
                    progress.Update((int)totalRowsProcessed);
                }
            }

            progress.Complete();
            Log($"  ✓ Normalized {totalRowsProcessed:N0} rows");
        }

        private double[] NormalizeColumn(string featureName, Array data, int start, int count,
            FeatureStatisticsCollector.FeatureStats? stats)
        {
            var normalizedData = new double[count];
            var normType = _config.GetNormalizationType(featureName);

            // If no statistics available, return original values
            if (stats == null)
            {
                Log($"  Warning: No statistics for {featureName}, keeping original values");
                for (int i = 0; i < count; i++)
                {
                    normalizedData[i] = Convert.ToDouble(data.GetValue(start + i));
                }
                return normalizedData;
            }

            // Apply normalization based on type
            for (int i = 0; i < count; i++)
            {
                double value = Convert.ToDouble(data.GetValue(start + i)!);
                normalizedData[i] = NormalizeValue(value, stats, normType);
            }

            return normalizedData;
        }

        private double NormalizeValue(double value, FeatureStatisticsCollector.FeatureStats stats,
            NormalizationType type)
        {
            // Handle invalid values
            if (double.IsNaN(value) || double.IsInfinity(value))
                return 0.0;

            switch (type)
            {
                case NormalizationType.None:
                    return value;

                case NormalizationType.StandardScaler:
                    // Z-score normalization: (x - mean) / std
                    if (stats.StdDev < 1e-10) return 0.0;
                    return (value - stats.Mean) / stats.StdDev;

                case NormalizationType.RobustScaler:
                    // Robust scaling: (x - median) / IQR
                    if (stats.IQR < 1e-10) return 0.0;
                    return (value - stats.Median) / stats.IQR;

                case NormalizationType.QuantileTransform:
                    // Map to uniform distribution [0, 1]
                    // Simplified: use min-max with clipping
                    if (stats.Max - stats.Min < 1e-10) return 0.5;
                    double normalized = (value - stats.Min) / (stats.Max - stats.Min);
                    return Math.Max(0, Math.Min(1, normalized));

                case NormalizationType.MinMaxScaler:
                    // Scale to [-1, 1]
                    if (stats.Max - stats.Min < 1e-10) return 0.0;
                    double minMaxNorm = (value - stats.Min) / (stats.Max - stats.Min);
                    return 2 * minMaxNorm - 1;

                default:
                    return value;
            }
        }

        private Array CopyIntArraySlice(Array source, int start, int count)
        {
            var result = new int[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = Convert.ToInt32(source.GetValue(start + i));
            }

            return result;
        }

        private Array CopyLongArraySlice(Array source, int start, int count)
        {
            var result = new long[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = Convert.ToInt64(source.GetValue(start + i));
            }

            return result;
        }

        private async Task WriteChunkAsync(ParquetWriter writer, ParquetSchema schema,
            Dictionary<string, Array> data, int rowCount)
        {
            using var rowGroup = writer.CreateRowGroup();

            foreach (var field in schema.DataFields)
            {
                var columnData = data[field.Name];
                var dataColumn = new DataColumn(field, columnData);
                await rowGroup.WriteColumnAsync(dataColumn);
            }
        }

        /// <summary>
        /// Validate normalization by checking value ranges
        /// </summary>
        public async Task ValidateNormalizationAsync(string normalizedPath)
        {
            Log("\nValidating normalized data...");

            using var inputStream = File.OpenRead(normalizedPath);
            using var reader = await ParquetReader.CreateAsync(inputStream);

            var schema = reader.Schema;
            var featureRanges = new Dictionary<string, (double min, double max, double mean)>();

            // Read first row group for validation
            using var rowGroupReader = reader.OpenRowGroupReader(0);

            foreach (var field in schema.DataFields)
            {
                if (field.Name == "label" || field.Name == "timestamp") continue;

                var column = await rowGroupReader.ReadColumnAsync(field);
                var values = new List<double>();

                for (int i = 0; i < Math.Min(1000, column.Data.Length); i++)
                {
                    values.Add(Convert.ToDouble(column.Data.GetValue(i)));
                }

                if (values.Count > 0)
                {
                    featureRanges[field.Name] = (values.Min(), values.Max(), values.Average());
                }
            }

            // Report validation results
            Log("\n  Feature ranges (sample of first 1000 rows):");
            Log("  " + new string('-', 80));

            var grouped = featureRanges.GroupBy(kvp => _config.GetNormalizationType(kvp.Key));

            foreach (var group in grouped)
            {
                Log($"\n  {group.Key} features:");
                foreach (var kvp in group.OrderBy(x => x.Key))
                {
                    var (min, max, mean) = kvp.Value;
                    Log($"    {kvp.Key,-40} Min: {min,8:F4}  Max: {max,8:F4}  Mean: {mean,8:F4}");
                }
            }
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
    }
}