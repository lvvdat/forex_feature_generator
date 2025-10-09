using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Normalization
{
    /// <summary>
    /// Comprehensive feature normalization system with appropriate methods for each feature type
    /// Optimized for 3-class prediction (-1, 0, 1)
    /// </summary>
    public class FeatureNormalizationSystem
    {
        private readonly Dictionary<string, NormalizationConfig> _featureConfigs;
        private readonly Dictionary<string, NormalizationStatistics> _statistics;
        private readonly RollingWindow<FeatureVector> _featureHistory;

        public FeatureNormalizationSystem(int historySize = 1000)
        {
            _featureConfigs = InitializeNormalizationConfigs();
            _statistics = new Dictionary<string, NormalizationStatistics>();
            _featureHistory = new RollingWindow<FeatureVector>(historySize);
        }

        /// <summary>
        /// Configuration for how each feature should be normalized
        /// </summary>
        public class NormalizationConfig
        {
            public string FeatureName { get; set; }
            public NormalizationType Type { get; set; }
            public double[] Parameters { get; set; }  // Method-specific parameters
            public bool ClipOutliers { get; set; }
            public double ClipMin { get; set; }
            public double ClipMax { get; set; }
            public string Description { get; set; }
        }

        /// <summary>
        /// Statistics needed for normalization
        /// </summary>
        public class NormalizationStatistics
        {
            public double Mean { get; set; }
            public double StdDev { get; set; }
            public double Median { get; set; }
            public double MAD { get; set; }  // Median Absolute Deviation
            public double Min { get; set; }
            public double Max { get; set; }
            public double Q1 { get; set; }   // 25th percentile
            public double Q3 { get; set; }   // 75th percentile
            public double IQR { get; set; }  // Interquartile range
            public List<double> HistoricalValues { get; set; }
            public DateTime LastUpdate { get; set; }
        }

        public enum NormalizationType
        {
            ZScore,           // (x - μ) / σ : For normally distributed features
            RobustZScore,     // (x - median) / MAD : For features with outliers
            MinMax,           // (x - min) / (max - min) : For bounded features
            Sigmoid,          // 2/(1+exp(-x)) - 1 : For unbounded features needing smooth bounds
            Tanh,             // tanh(x) : Alternative to sigmoid
            Percentile,       // Rank-based : For features where relative position matters
            Log,              // log(1 + |x|) * sign(x) : For skewed distributions
            Reciprocal,       // 1/x : For inverse relationships
            Clipping,         // clip(x, min, max) : Direct clipping
            None,             // No normalization (already normalized)
            Adaptive          // Dynamically chosen based on distribution
        }

        /// <summary>
        /// Initialize normalization configurations for all feature types
        /// </summary>
        private Dictionary<string, NormalizationConfig> InitializeNormalizationConfigs()
        {
            var configs = new Dictionary<string, NormalizationConfig>();

            // ===== DIRECTIONAL FEATURES (Already optimized, minimal normalization) =====

            // Momentum features - Z-score normalization
            AddConfig(configs, "dir_momentum_z5", NormalizationType.Clipping, -3, 3,
                "Already z-scored, just clip extremes");
            AddConfig(configs, "dir_momentum_z10", NormalizationType.Clipping, -3, 3);
            AddConfig(configs, "dir_momentum_accel", NormalizationType.None,
                "Already sigmoid normalized");
            AddConfig(configs, "dir_momentum_quality", NormalizationType.None,
                "Already in [0,1] range");

            // Price action features - Various methods
            AddConfig(configs, "dir_candle_direction", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "dir_pattern_strength", NormalizationType.None,
                "Already sigmoid normalized");
            AddConfig(configs, "dir_price_position", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "dir_hhll_signal", NormalizationType.None,
                "Discrete signal {-1,0,1}");

            // Volume features - Mostly normalized
            AddConfig(configs, "dir_volume_direction", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "dir_volume_pressure", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "dir_vol_mom_correlation", NormalizationType.None,
                "Correlation coefficient [-1,1]");

            // Composite signals - Already normalized
            AddConfig(configs, "dir_composite_primary", NormalizationType.None,
                "Composite signal {-1,0,1}");
            AddConfig(configs, "dir_probability", NormalizationType.None,
                "Already sigmoid normalized");
            AddConfig(configs, "dir_confidence", NormalizationType.Clipping, 0, 1,
                "Confidence score [0,1]");

            // ===== TECHNICAL INDICATORS (Need specific normalization) =====

            // RSI - Already bounded [0,100], convert to [-1,1]
            AddConfig(configs, "tech_rsi_normalized", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "tech_rsi_signal", NormalizationType.None,
                "Discrete signal");
            AddConfig(configs, "tech_rsi_momentum", NormalizationType.None,
                "Already sigmoid normalized");

            // MACD - Normalize by ATR or z-score
            AddConfig(configs, "tech_macd_normalized", NormalizationType.None,
                "Already normalized by ATR");
            AddConfig(configs, "tech_macd_cross", NormalizationType.None,
                "Discrete signal {-1,0,1}");
            AddConfig(configs, "tech_macd_quality", NormalizationType.None,
                "Quality score [0,1]");

            // Stochastic - Convert from [0,100] to [-1,1]
            AddConfig(configs, "tech_stoch_normalized", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "tech_stoch_cross", NormalizationType.None,
                "Discrete signal");

            // Bollinger Bands
            AddConfig(configs, "tech_bb_position", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "tech_bb_squeeze", NormalizationType.None,
                "Binary signal {0,1}");
            AddConfig(configs, "tech_bb_touch", NormalizationType.None,
                "Discrete signal {-1,0,1}");

            // Moving Averages
            AddConfig(configs, "tech_ma_alignment", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "tech_ma_dev_9", NormalizationType.None,
                "Already sigmoid normalized");

            // Volatility
            AddConfig(configs, "tech_atr_ratio", NormalizationType.Sigmoid, -2.0, 2.0,
                "ATR ratio, sigmoid with steepness 2");
            AddConfig(configs, "tech_vol_percentile", NormalizationType.None,
                "Already percentile [0,1]");

            // ===== MICROSTRUCTURE FEATURES (Complex normalization) =====

            // Order flow - Already normalized
            AddConfig(configs, "micro_flow_imbalance", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "micro_cvd_normalized", NormalizationType.None,
                "Already sigmoid normalized");
            AddConfig(configs, "micro_vwof", NormalizationType.Sigmoid, -0.3, 3.0,
                "Volume-weighted flow, sigmoid");

            // Spread features - Z-score or robust scaling
            AddConfig(configs, "micro_spread_zscore", NormalizationType.Clipping, -3, 3,
                "Already z-scored, clip extremes");
            AddConfig(configs, "micro_spread_regime", NormalizationType.None,
                "Discrete regime {-1,0,1}");
            AddConfig(configs, "micro_effective_spread", NormalizationType.Log,
                "Log transform for skewed distribution");

            // Tick dynamics
            AddConfig(configs, "micro_tick_intensity", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "micro_tick_direction", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "micro_tick_clustering", NormalizationType.None,
                "Already normalized [0,1]");

            // VWAP features
            AddConfig(configs, "micro_vwap_deviation", NormalizationType.None,
                "Already sigmoid normalized");
            AddConfig(configs, "micro_vwap_pull", NormalizationType.None,
                "Already normalized [-1,1]");

            // Market depth
            AddConfig(configs, "micro_kyle_lambda", NormalizationType.None,
                "Already sigmoid normalized");
            AddConfig(configs, "micro_amihud_illiquidity", NormalizationType.None,
                "Already sigmoid normalized");

            // ===== REGIME FEATURES (Context-dependent normalization) =====

            // Regime identification
            AddConfig(configs, "regime_type", NormalizationType.None,
                "Discrete regime {0,1,2}");
            AddConfig(configs, "regime_confidence", NormalizationType.None,
                "Confidence score [0,1]");
            AddConfig(configs, "regime_directional_bias", NormalizationType.None,
                "Already normalized [-1,1]");

            // Volatility regime
            AddConfig(configs, "vol_regime_type", NormalizationType.None,
                "Discrete regime {-1,0,1}");
            AddConfig(configs, "vol_trend", NormalizationType.None,
                "Already sigmoid normalized");
            AddConfig(configs, "vol_garch_forecast", NormalizationType.RobustZScore,
                "GARCH volatility, robust z-score");

            // Trend features
            AddConfig(configs, "trend_mtf_alignment", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "trend_efficiency", NormalizationType.None,
                "Efficiency ratio [0,1]");
            AddConfig(configs, "trend_quality", NormalizationType.None,
                "Quality score [0,1]");

            // Market stress
            AddConfig(configs, "market_stress", NormalizationType.None,
                "Already sigmoid normalized");
            AddConfig(configs, "risk_sentiment", NormalizationType.None,
                "Discrete sentiment {-1,0,1}");

            // Fractal/Chaos
            AddConfig(configs, "fractal_dimension", NormalizationType.None,
                "Already normalized around 1.5");
            AddConfig(configs, "hurst_exponent", NormalizationType.None,
                "Already normalized [-1,1]");
            AddConfig(configs, "chaos_indicator", NormalizationType.None,
                "Already sigmoid normalized");

            return configs;
        }

        /// <summary>
        /// Helper to add configuration
        /// </summary>
        private void AddConfig(Dictionary<string, NormalizationConfig> configs,
            string featureName, NormalizationType type,
            double param1 = 0, double param2 = 0, string description = "")
        {
            configs[featureName] = new NormalizationConfig
            {
                FeatureName = featureName,
                Type = type,
                Parameters = new[] { param1, param2 },
                ClipOutliers = type == NormalizationType.Clipping,
                ClipMin = type == NormalizationType.Clipping ? param1 : -10,
                ClipMax = type == NormalizationType.Clipping ? param2 : 10,
                Description = description
            };
        }

        private void AddConfig(Dictionary<string, NormalizationConfig> configs,
            string featureName, NormalizationType type, string description)
        {
            AddConfig(configs, featureName, type, 0, 0, description);
        }

        /// <summary>
        /// Main normalization method - normalizes entire feature vector
        /// </summary>
        public FeatureVector NormalizeFeatures(FeatureVector features)
        {
            // Update statistics with new data
            UpdateStatistics(features);

            var normalizedFeatures = new FeatureVector
            {
                Timestamp = features.Timestamp
            };

            foreach (var kvp in features.Features)
            {
                var featureName = kvp.Key;
                var value = kvp.Value;

                // Get normalization config
                if (!_featureConfigs.TryGetValue(featureName, out var config))
                {
                    // Default to adaptive normalization for unknown features
                    config = new NormalizationConfig
                    {
                        FeatureName = featureName,
                        Type = NormalizationType.Adaptive,
                        ClipOutliers = true,
                        ClipMin = -5,
                        ClipMax = 5
                    };
                }

                // Get statistics
                if (!_statistics.TryGetValue(featureName, out var stats))
                {
                    // Initialize statistics for new feature
                    stats = InitializeStatistics(featureName);
                }

                // Apply normalization
                var normalizedValue = ApplyNormalization(value, config, stats);

                // Final safety clipping to prevent extreme values
                normalizedValue = Math.Max(-3, Math.Min(3, normalizedValue));

                normalizedFeatures.AddFeature(featureName, normalizedValue);
            }

            // Store in history for adaptive statistics
            _featureHistory.Add(normalizedFeatures);

            return normalizedFeatures;
        }

        /// <summary>
        /// Apply specific normalization method
        /// </summary>
        private double ApplyNormalization(double value, NormalizationConfig config,
            NormalizationStatistics stats)
        {
            // Handle outliers first if configured
            if (config.ClipOutliers && config.Type != NormalizationType.Clipping)
            {
                value = Math.Max(config.ClipMin, Math.Min(config.ClipMax, value));
            }

            switch (config.Type)
            {
                case NormalizationType.ZScore:
                    return ApplyZScore(value, stats.Mean, stats.StdDev);

                case NormalizationType.RobustZScore:
                    return ApplyRobustZScore(value, stats.Median, stats.MAD);

                case NormalizationType.MinMax:
                    return ApplyMinMax(value, stats.Min, stats.Max);

                case NormalizationType.Sigmoid:
                    var steepness = config.Parameters[0] > 0 ? config.Parameters[0] : 1.0;
                    return ApplySigmoid(value, steepness);

                case NormalizationType.Tanh:
                    var scale = config.Parameters[0] > 0 ? config.Parameters[0] : 1.0;
                    return Math.Tanh(value * scale);

                case NormalizationType.Percentile:
                    return ApplyPercentile(value, stats.HistoricalValues);

                case NormalizationType.Log:
                    return ApplyLogTransform(value);

                case NormalizationType.Reciprocal:
                    return ApplyReciprocal(value);

                case NormalizationType.Clipping:
                    return Math.Max(config.Parameters[0], Math.Min(config.Parameters[1], value));

                case NormalizationType.Adaptive:
                    return ApplyAdaptiveNormalization(value, stats);

                case NormalizationType.None:
                default:
                    return value;
            }
        }

        /// <summary>
        /// Z-score normalization: (x - μ) / σ
        /// Good for normally distributed features
        /// </summary>
        private double ApplyZScore(double value, double mean, double stdDev)
        {
            if (stdDev < 1e-10) return 0;
            return (value - mean) / stdDev;
        }

        /// <summary>
        /// Robust Z-score: (x - median) / MAD
        /// Good for features with outliers
        /// </summary>
        private double ApplyRobustZScore(double value, double median, double mad)
        {
            if (mad < 1e-10) return 0;
            return (value - median) / (1.4826 * mad);  // 1.4826 makes MAD consistent with std dev
        }

        /// <summary>
        /// Min-Max normalization: (x - min) / (max - min) * 2 - 1
        /// Maps to [-1, 1] for directional features
        /// </summary>
        private double ApplyMinMax(double value, double min, double max)
        {
            if (max - min < 1e-10) return 0;
            return 2 * (value - min) / (max - min) - 1;
        }

        /// <summary>
        /// Sigmoid normalization: 2/(1 + exp(-steepness*x)) - 1
        /// Smooth mapping to [-1, 1]
        /// </summary>
        private double ApplySigmoid(double value, double steepness = 1.0)
        {
            return 2.0 / (1.0 + Math.Exp(-steepness * value)) - 1.0;
        }

        /// <summary>
        /// Percentile rank normalization
        /// Good for features where relative position matters
        /// </summary>
        private double ApplyPercentile(double value, List<double> historicalValues)
        {
            if (historicalValues == null || historicalValues.Count < 10)
                return 0;

            var rank = historicalValues.Count(v => v < value);
            var percentile = (double)rank / historicalValues.Count;
            return 2 * percentile - 1;  // Map to [-1, 1]
        }

        /// <summary>
        /// Log transformation: log(1 + |x|) * sign(x)
        /// Good for skewed distributions
        /// </summary>
        private double ApplyLogTransform(double value)
        {
            return Math.Log(1 + Math.Abs(value)) * Math.Sign(value);
        }

        /// <summary>
        /// Reciprocal transformation: 2/(1 + |x|) - 1
        /// Good for inverse relationships
        /// </summary>
        private double ApplyReciprocal(double value)
        {
            return 2.0 / (1.0 + Math.Abs(value)) - 1.0;
        }

        /// <summary>
        /// Adaptive normalization based on distribution characteristics
        /// </summary>
        private double ApplyAdaptiveNormalization(double value, NormalizationStatistics stats)
        {
            // Check distribution characteristics
            var isNormal = CheckNormality(stats);
            var hasOutliers = CheckOutliers(stats);
            var isSkewed = CheckSkewness(stats);

            if (isNormal && !hasOutliers)
            {
                // Use standard z-score for normal distributions
                return ApplyZScore(value, stats.Mean, stats.StdDev);
            }
            else if (hasOutliers)
            {
                // Use robust z-score for distributions with outliers
                return ApplyRobustZScore(value, stats.Median, stats.MAD);
            }
            else if (isSkewed)
            {
                // Use log transform for skewed distributions
                return ApplyLogTransform(value);
            }
            else
            {
                // Default to sigmoid for unknown distributions
                return ApplySigmoid(value);
            }
        }

        /// <summary>
        /// Update statistics for adaptive normalization
        /// </summary>
        private void UpdateStatistics(FeatureVector features)
        {
            foreach (var kvp in features.Features)
            {
                var featureName = kvp.Key;
                var value = kvp.Value;

                if (!_statistics.ContainsKey(featureName))
                {
                    _statistics[featureName] = InitializeStatistics(featureName);
                }

                var stats = _statistics[featureName];

                // Add to historical values (rolling window)
                if (stats.HistoricalValues.Count >= 1000)
                {
                    stats.HistoricalValues.RemoveAt(0);
                }
                stats.HistoricalValues.Add(value);

                // Update statistics every 100 samples for efficiency
                if (stats.HistoricalValues.Count % 100 == 0)
                {
                    UpdateDetailedStatistics(stats);
                }
            }
        }

        /// <summary>
        /// Initialize statistics for a new feature
        /// </summary>
        private NormalizationStatistics InitializeStatistics(string featureName)
        {
            return new NormalizationStatistics
            {
                Mean = 0,
                StdDev = 1,
                Median = 0,
                MAD = 1,
                Min = -1,
                Max = 1,
                Q1 = -0.25,
                Q3 = 0.25,
                IQR = 0.5,
                HistoricalValues = new List<double>(),
                LastUpdate = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Update detailed statistics
        /// </summary>
        private void UpdateDetailedStatistics(NormalizationStatistics stats)
        {
            if (stats.HistoricalValues.Count < 10) return;

            var values = stats.HistoricalValues.ToArray();
            var sorted = values.OrderBy(v => v).ToArray();

            // Basic statistics
            stats.Mean = values.Average();
            stats.StdDev = Math.Sqrt(values.Select(v => Math.Pow(v - stats.Mean, 2)).Average());
            stats.Min = sorted.First();
            stats.Max = sorted.Last();

            // Robust statistics
            stats.Median = GetPercentile(sorted, 0.50);
            stats.Q1 = GetPercentile(sorted, 0.25);
            stats.Q3 = GetPercentile(sorted, 0.75);
            stats.IQR = stats.Q3 - stats.Q1;

            // MAD (Median Absolute Deviation)
            var deviations = values.Select(v => Math.Abs(v - stats.Median)).OrderBy(d => d).ToArray();
            stats.MAD = GetPercentile(deviations, 0.50);

            stats.LastUpdate = DateTime.UtcNow;
        }

        /// <summary>
        /// Get percentile value from sorted array
        /// </summary>
        private double GetPercentile(double[] sorted, double percentile)
        {
            var index = percentile * (sorted.Length - 1);
            var lower = (int)Math.Floor(index);
            var upper = (int)Math.Ceiling(index);
            var weight = index - lower;

            if (lower == upper) return sorted[lower];
            return sorted[lower] * (1 - weight) + sorted[upper] * weight;
        }

        /// <summary>
        /// Check if distribution is approximately normal
        /// </summary>
        private bool CheckNormality(NormalizationStatistics stats)
        {
            if (stats.HistoricalValues.Count < 30) return false;

            // Simple normality check using z-scores
            var zScores = stats.HistoricalValues
                .Select(v => ApplyZScore(v, stats.Mean, stats.StdDev))
                .ToArray();

            // Check if 68% of values are within 1 std dev
            var within1Std = zScores.Count(z => Math.Abs(z) <= 1) / (double)zScores.Length;

            // Check if 95% of values are within 2 std dev
            var within2Std = zScores.Count(z => Math.Abs(z) <= 2) / (double)zScores.Length;

            return within1Std > 0.60 && within1Std < 0.75 &&
                   within2Std > 0.90 && within2Std < 0.98;
        }

        /// <summary>
        /// Check for outliers using IQR method
        /// </summary>
        private bool CheckOutliers(NormalizationStatistics stats)
        {
            if (stats.HistoricalValues.Count < 30) return false;

            var lowerBound = stats.Q1 - 1.5 * stats.IQR;
            var upperBound = stats.Q3 + 1.5 * stats.IQR;

            var outlierRatio = stats.HistoricalValues
                .Count(v => v < lowerBound || v > upperBound) / (double)stats.HistoricalValues.Count;

            return outlierRatio > 0.05;  // More than 5% outliers
        }

        /// <summary>
        /// Check for skewness
        /// </summary>
        private bool CheckSkewness(NormalizationStatistics stats)
        {
            if (stats.HistoricalValues.Count < 30) return false;

            // Calculate skewness
            var values = stats.HistoricalValues.ToArray();
            var n = values.Length;
            var mean = stats.Mean;
            var stdDev = stats.StdDev;

            if (stdDev < 1e-10) return false;

            var skewness = values.Sum(v => Math.Pow((v - mean) / stdDev, 3)) * n /
                          ((n - 1) * (n - 2));

            return Math.Abs(skewness) > 1.0;  // Significantly skewed
        }

        /// <summary>
        /// Get normalization configuration for a feature
        /// </summary>
        public NormalizationConfig GetFeatureConfig(string featureName)
        {
            return _featureConfigs.TryGetValue(featureName, out var config) ? config : null;
        }

        /// <summary>
        /// Get current statistics for a feature
        /// </summary>
        public NormalizationStatistics GetFeatureStatistics(string featureName)
        {
            return _statistics.TryGetValue(featureName, out var stats) ? stats : null;
        }

        /// <summary>
        /// Export normalization parameters for model deployment
        /// </summary>
        public Dictionary<string, object> ExportNormalizationParameters()
        {
            var parameters = new Dictionary<string, object>();

            foreach (var kvp in _statistics)
            {
                var featureName = kvp.Key;
                var stats = kvp.Value;
                var config = _featureConfigs.TryGetValue(featureName, out var c) ? c : null;

                parameters[featureName] = new
                {
                    Type = config?.Type.ToString() ?? "Adaptive",
                    Mean = stats.Mean,
                    StdDev = stats.StdDev,
                    Median = stats.Median,
                    MAD = stats.MAD,
                    Min = stats.Min,
                    Max = stats.Max,
                    Q1 = stats.Q1,
                    Q3 = stats.Q3,
                    Parameters = config?.Parameters ?? new double[0]
                };
            }

            return parameters;
        }
    }
}