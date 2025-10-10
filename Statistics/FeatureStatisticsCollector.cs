using System.Collections.Concurrent;

namespace ForexFeatureGenerator.Core.Statistics
{
    /// <summary>
    /// Collects statistics for each feature during generation for later normalization
    /// Uses memory-efficient incremental algorithms
    /// </summary>
    public class FeatureStatisticsCollector
    {
        private readonly ConcurrentDictionary<string, FeatureStats> _featureStats = new();
        private readonly object _lockObj = new();

        public class FeatureStats
        {
            // For mean and standard deviation (Welford's algorithm)
            public long Count { get; set; }
            public double Mean { get; set; }
            public double M2 { get; set; } // Sum of squares of differences from mean

            // For min/max
            public double Min { get; set; } = double.MaxValue;
            public double Max { get; set; } = double.MinValue;

            // For median and quantiles (reservoir sampling)
            public List<double> Reservoir { get; set; } = new(10000);
            private readonly Random _random = new();

            // For robust scaler (median and IQR)
            public double Median { get; set; }
            public double Q1 { get; set; }
            public double Q3 { get; set; }

            // Computed properties
            public double Variance => Count > 1 ? M2 / (Count - 1) : 0;
            public double StdDev => Math.Sqrt(Variance);
            public double IQR => Q3 - Q1;

            public void UpdateIncremental(double value)
            {
                Count++;

                // Welford's algorithm for mean and variance
                double delta = value - Mean;
                Mean += delta / Count;
                double delta2 = value - Mean;
                M2 += delta * delta2;

                // Update min/max
                Min = Math.Min(Min, value);
                Max = Math.Max(Max, value);

                // Reservoir sampling for quantiles (keep 10000 samples)
                if (Reservoir.Count < 10000)
                {
                    Reservoir.Add(value);
                }
                else
                {
                    int j = _random.Next((int)Count);
                    if (j < 10000)
                    {
                        Reservoir[j] = value;
                    }
                }
            }

            public void ComputeQuantiles()
            {
                if (Reservoir.Count == 0) return;

                var sorted = Reservoir.OrderBy(x => x).ToList();
                int n = sorted.Count;

                Median = GetPercentile(sorted, 0.50);
                Q1 = GetPercentile(sorted, 0.25);
                Q3 = GetPercentile(sorted, 0.75);
            }

            private double GetPercentile(List<double> sorted, double percentile)
            {
                int n = sorted.Count;
                double index = percentile * (n - 1);
                int lower = (int)Math.Floor(index);
                int upper = (int)Math.Ceiling(index);

                if (lower == upper) return sorted[lower];

                double weight = index - lower;
                return sorted[lower] * (1 - weight) + sorted[upper] * weight;
            }
        }

        public void UpdateFeature(string featureName, double value)
        {
            // Skip invalid values
            if (double.IsNaN(value) || double.IsInfinity(value)) return;

            var stats = _featureStats.GetOrAdd(featureName, _ => new FeatureStats());

            lock (stats)
            {
                stats.UpdateIncremental(value);
            }
        }

        public void UpdateFeatures(Dictionary<string, double> features)
        {
            foreach (var kvp in features)
            {
                UpdateFeature(kvp.Key, kvp.Value);
            }
        }

        public void FinalizeStatistics()
        {
            Parallel.ForEach(_featureStats.Values, stats =>
            {
                stats.ComputeQuantiles();
            });
        }

        public Dictionary<string, FeatureStats> GetStatistics()
        {
            return _featureStats.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }

        public void SaveStatistics(string path)
        {
            FinalizeStatistics();

            using var writer = new StreamWriter(path);
            writer.WriteLine("Feature,Count,Mean,StdDev,Min,Max,Q1,Median,Q3,IQR");

            foreach (var kvp in _featureStats.OrderBy(x => x.Key))
            {
                var stats = kvp.Value;
                writer.WriteLine($"{kvp.Key},{stats.Count},{stats.Mean:F6},{stats.StdDev:F6}," +
                               $"{stats.Min:F6},{stats.Max:F6},{stats.Q1:F6},{stats.Median:F6}," +
                               $"{stats.Q3:F6},{stats.IQR:F6}");
            }
        }

        public static FeatureStatisticsCollector LoadStatistics(string path)
        {
            var collector = new FeatureStatisticsCollector();

            using var reader = new StreamReader(path);
            reader.ReadLine(); // Skip header

            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                var parts = line.Split(',');
                if (parts.Length < 10) continue;

                var stats = new FeatureStats
                {
                    Count = long.Parse(parts[1]),
                    Mean = double.Parse(parts[2]),
                    Min = double.Parse(parts[4]),
                    Max = double.Parse(parts[5]),
                    Q1 = double.Parse(parts[6]),
                    Median = double.Parse(parts[7]),
                    Q3 = double.Parse(parts[8])
                };

                // Reconstruct M2 from StdDev
                double stdDev = double.Parse(parts[3]);
                stats.M2 = stdDev * stdDev * (stats.Count - 1);

                collector._featureStats[parts[0]] = stats;
            }

            return collector;
        }
    }
}