using System.Text.Json;
using System.Text.Json.Serialization;

namespace ForexFeatureGenerator.Normalization
{
    /// <summary>
    /// Comprehensive feature normalizer for ForexFeatureGenerator
    /// Handles 200+ features with appropriate normalization strategies
    /// Designed for both training and production use
    /// </summary>
    public class ComprehensiveFeatureNormalizer
    {
        #region Enums and Classes

        /// <summary>
        /// Normalization type for each feature category
        /// </summary>
        public enum NormalizationType
        {
            None,           // No normalization (already in proper range)
            Robust,         // Median and IQR based (resistant to outliers)
            Standard,       // Mean and StdDev based (Z-score)
            MinMax,         // Scale to [0,1] range
            LogRobust,      // Log transform + Robust scaling
            LogStandard,    // Log transform + Standard scaling
            Sigmoid,        // Sigmoid transformation for extreme values
            Tanh            // Tanh transformation for [-1,1] range
        }

        /// <summary>
        /// Base class for all scalers
        /// </summary>
        public abstract class Scaler
        {
            public abstract void Fit(double[] values);
            public abstract double Transform(double value);
            public abstract double InverseTransform(double value);
            public abstract Dictionary<string, double> GetParameters();
            public abstract void SetParameters(Dictionary<string, double> parameters);
        }

        #endregion

        #region Scaler Implementations

        /// <summary>
        /// Robust scaler using median and IQR
        /// Best for features with outliers (price, volatility)
        /// Formula: (x - median) / IQR
        /// </summary>
        public class RobustScaler : Scaler
        {
            private double _median;
            private double _q1;
            private double _q3;
            private double _iqr;

            public override void Fit(double[] values)
            {
                if (values == null || values.Length == 0)
                    throw new ArgumentException("Values cannot be null or empty");

                var sorted = values.OrderBy(v => v).ToArray();
                int n = sorted.Length;

                // Calculate median
                _median = n % 2 == 0
                    ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                    : sorted[n / 2];

                // Calculate Q1 (25th percentile) and Q3 (75th percentile)
                int q1Index = (int)Math.Floor(n * 0.25);
                int q3Index = (int)Math.Floor(n * 0.75);

                _q1 = sorted[q1Index];
                _q3 = sorted[q3Index];
                _iqr = _q3 - _q1;

                // Handle case where IQR is zero (all values are the same)
                if (Math.Abs(_iqr) < 1e-10)
                {
                    // Use mean absolute deviation from median as fallback
                    var mad = values.Select(v => Math.Abs(v - _median)).Average();
                    _iqr = mad > 0 ? mad * 1.4826 : 1.0; // 1.4826 is the constant to make MAD consistent with std dev
                }
            }

            public override double Transform(double value)
            {
                return (value - _median) / _iqr;
            }

            public override double InverseTransform(double value)
            {
                return value * _iqr + _median;
            }

            public override Dictionary<string, double> GetParameters()
            {
                return new Dictionary<string, double>
                {
                    ["median"] = _median,
                    ["q1"] = _q1,
                    ["q3"] = _q3,
                    ["iqr"] = _iqr
                };
            }

            public override void SetParameters(Dictionary<string, double> parameters)
            {
                _median = parameters["median"];
                _q1 = parameters["q1"];
                _q3 = parameters["q3"];
                _iqr = parameters["iqr"];
            }
        }

        /// <summary>
        /// Standard scaler using mean and standard deviation
        /// Best for normally distributed features
        /// Formula: (x - mean) / stddev
        /// </summary>
        public class StandardScaler : Scaler
        {
            private double _mean;
            private double _stdDev;

            public override void Fit(double[] values)
            {
                if (values == null || values.Length == 0)
                    throw new ArgumentException("Values cannot be null or empty");

                _mean = values.Average();

                // Calculate standard deviation with Bessel's correction (n-1)
                double sumSquares = values.Sum(v => Math.Pow(v - _mean, 2));
                _stdDev = Math.Sqrt(sumSquares / (values.Length - 1));

                // Handle case where std dev is zero
                if (Math.Abs(_stdDev) < 1e-10)
                    _stdDev = 1.0;
            }

            public override double Transform(double value)
            {
                return (value - _mean) / _stdDev;
            }

            public override double InverseTransform(double value)
            {
                return value * _stdDev + _mean;
            }

            public override Dictionary<string, double> GetParameters()
            {
                return new Dictionary<string, double>
                {
                    ["mean"] = _mean,
                    ["stdDev"] = _stdDev
                };
            }

            public override void SetParameters(Dictionary<string, double> parameters)
            {
                _mean = parameters["mean"];
                _stdDev = parameters["stdDev"];
            }
        }

        /// <summary>
        /// MinMax scaler to scale features to [0,1] range
        /// Best for bounded features where relative position matters
        /// Formula: (x - min) / (max - min)
        /// </summary>
        public class MinMaxScaler : Scaler
        {
            private double _min;
            private double _max;
            private double _range;

            public override void Fit(double[] values)
            {
                if (values == null || values.Length == 0)
                    throw new ArgumentException("Values cannot be null or empty");

                _min = values.Min();
                _max = values.Max();
                _range = _max - _min;

                // Handle case where range is zero
                if (Math.Abs(_range) < 1e-10)
                    _range = 1.0;
            }

            public override double Transform(double value)
            {
                return (value - _min) / _range;
            }

            public override double InverseTransform(double value)
            {
                return value * _range + _min;
            }

            public override Dictionary<string, double> GetParameters()
            {
                return new Dictionary<string, double>
                {
                    ["min"] = _min,
                    ["max"] = _max,
                    ["range"] = _range
                };
            }

            public override void SetParameters(Dictionary<string, double> parameters)
            {
                _min = parameters["min"];
                _max = parameters["max"];
                _range = parameters["range"];
            }
        }

        /// <summary>
        /// Log-Robust scaler: applies log transform then robust scaling
        /// Best for skewed volume data with outliers
        /// </summary>
        public class LogRobustScaler : Scaler
        {
            private readonly RobustScaler _robustScaler = new RobustScaler();

            public override void Fit(double[] values)
            {
                var logValues = values.Select(ApplyLogTransform).ToArray();
                _robustScaler.Fit(logValues);
            }

            public override double Transform(double value)
            {
                var logValue = ApplyLogTransform(value);
                return _robustScaler.Transform(logValue);
            }

            public override double InverseTransform(double value)
            {
                var robustValue = _robustScaler.InverseTransform(value);
                return ApplyInverseLogTransform(robustValue);
            }

            /// <summary>
            /// Sign-preserving log transform
            /// Handles negative values properly
            /// </summary>
            private static double ApplyLogTransform(double value)
            {
                // For small values near zero, return 0
                if (Math.Abs(value) < 1e-10)
                    return 0;

                // Sign-preserving log transform: sign(x) * log(|x| + 1)
                return Math.Sign(value) * Math.Log(Math.Abs(value) + 1);
            }

            /// <summary>
            /// Inverse of sign-preserving log transform
            /// </summary>
            private static double ApplyInverseLogTransform(double value)
            {
                // sign(x) * (exp(|x|) - 1)
                return Math.Sign(value) * (Math.Exp(Math.Abs(value)) - 1);
            }

            public override Dictionary<string, double> GetParameters()
            {
                return _robustScaler.GetParameters();
            }

            public override void SetParameters(Dictionary<string, double> parameters)
            {
                _robustScaler.SetParameters(parameters);
            }
        }

        /// <summary>
        /// Log-Standard scaler: applies log transform then standard scaling
        /// Best for log-normal distributed data
        /// </summary>
        public class LogStandardScaler : Scaler
        {
            private readonly StandardScaler _standardScaler = new StandardScaler();

            public override void Fit(double[] values)
            {
                var logValues = values.Select(ApplyLogTransform).ToArray();
                _standardScaler.Fit(logValues);
            }

            public override double Transform(double value)
            {
                var logValue = ApplyLogTransform(value);
                return _standardScaler.Transform(logValue);
            }

            public override double InverseTransform(double value)
            {
                var standardValue = _standardScaler.InverseTransform(value);
                return ApplyInverseLogTransform(standardValue);
            }

            private static double ApplyLogTransform(double value)
            {
                if (Math.Abs(value) < 1e-10)
                    return 0;
                return Math.Sign(value) * Math.Log(Math.Abs(value) + 1);
            }

            private static double ApplyInverseLogTransform(double value)
            {
                return Math.Sign(value) * (Math.Exp(Math.Abs(value)) - 1);
            }

            public override Dictionary<string, double> GetParameters()
            {
                return _standardScaler.GetParameters();
            }

            public override void SetParameters(Dictionary<string, double> parameters)
            {
                _standardScaler.SetParameters(parameters);
            }
        }

        #endregion

        #region Feature Categorization

        private readonly Dictionary<string, NormalizationType> _featureNormalizationMap;
        private readonly Dictionary<string, Scaler> _scalers;
        private bool _isFitted = false;

        public ComprehensiveFeatureNormalizer()
        {
            _featureNormalizationMap = InitializeFeatureMap();
            _scalers = new Dictionary<string, Scaler>();
        }

        /// <summary>
        /// Initialize the complete feature normalization mapping
        /// Based on careful analysis of each feature's characteristics
        /// </summary>
        private Dictionary<string, NormalizationType> InitializeFeatureMap()
        {
            var map = new Dictionary<string, NormalizationType>();

            // ========================================
            // CATEGORY 1: NO NORMALIZATION NEEDED
            // Features already in proper range [0,1] or [0,100]
            // ========================================

            // RSI indicators (bounded 0-100, already normalized)
            AddFeatures(map, NormalizationType.None, new[]
            {
                "fg1_rsi_9",
                "fg1_rsi_14",  // M1 RSI
                "fg3_rsi_9",
                "fg3_rsi_14",
                "fg3_rsi_21",  // M5 RSI
                "fg3_rsi_oversold",
                "fg3_rsi_overbought"  // RSI binary flags
            });

            // Stochastic oscillators (bounded 0-100)
            AddFeatures(map, NormalizationType.None, new[]
            {
                "fg1_stoch_k",
                "fg1_stoch_d",  // M1 stochastic
                "fg3_stoch_k_14",
                "fg3_stoch_d_14"  // M5 stochastic
            });

            // Aroon indicators (bounded 0-100)
            AddFeatures(map, NormalizationType.None, new[]
            {
                "fg1_aroon_up",
                "fg1_aroon_down",
                "fg1_aroon_osc"
            });

            // Percentages and normalized ratios (already 0-1 or 0-100)
            AddFeatures(map, NormalizationType.None, new[]
            {
                // M1 percentages
                "fg1_up_volume_pct",
                "fg1_down_volume_pct",
                "fg1_up_price_pct",
                "fg1_down_price_pct",
                "fg1_range_pct",
                "fg1_bb_pct",
                "fg1_atr_pct",
                "fg1_upper_wick",
                "fg1_lower_wick",
                "fg1_body_size",
                "fg1_normalized_range",
                "fg1_ema_ratio",
                "fg1_volume_imbalance",
                
                // Order flow ratios (already normalized -1 to 1)
                "fg2_of_buy_sell_ratio",
                "fg2_of_pressure_ratio",
                "fg2_of_aggressive_ratio",
                "fg2_of_quote_imbalance",
                "fg2_of_bid_depth_change",
                "fg2_of_ask_depth_change",
                "fg2_of_book_imbalance",
                "fg2_of_flow_autocorr",
                
                // Liquidity ratios
                "fg2_liquidity_resilience",
                "fg2_liquidity_tick_clustering",
                
                // Market regime (categorical and confidence values)
                "fg2_regime_type",
                "fg2_regime_confidence",
                "fg2_regime_transition_prob",
                "fg2_efficiency_ratio",
                "fg2_variance_ratio",
                "fg2_jump_intensity"
            });

            // Binary pattern indicators (0 or 1)
            AddFeatures(map, NormalizationType.None, new[]
            {
                // Candlestick patterns
                "fg2_pattern_bullish_engulfing",
                "fg2_pattern_bearish_engulfing",
                "fg2_pattern_hammer",
                "fg2_pattern_shooting_star",
                "fg2_pattern_doji",
                "fg2_pattern_three_white_soldiers",
                "fg2_pattern_three_black_crows",
                "fg2_pattern_morning_star",
                "fg2_pattern_evening_star",
                "fg2_pattern_bullish_harami",
                "fg2_pattern_bearish_harami",
                "fg2_pattern_tweezer_top",
                "fg2_pattern_tweezer_bottom",
                "fg2_pattern_spinning_top",
                "fg2_pattern_marubozu",
                
                // Price action patterns
                "fg2_pattern_higher_high",
                "fg2_pattern_lower_low",
                "fg2_pattern_head_shoulders",
                "fg2_pattern_inverse_head_shoulders",
                "fg2_pattern_flag",
                "fg2_pattern_wedge",
                "fg2_pattern_confirmation",
                "fg2_pattern_success_rate",
                
                // Pattern metrics
                "fg2_pattern_strength",
                "fg2_pattern_frequency"
            });

            // M5 bounded indicators
            AddFeatures(map, NormalizationType.None, new[]
            {
                "fg3_bb_position",
                "fg3_bb_percent_b",
                "fg3_bb_squeeze",
                "fg3_range_expansion",
                "fg3_range_contraction",
                "fg3_volume_ratio",
                "fg3_obv_divergence",
                "fg3_pvi",
                "fg3_nvi",
                "fg3_mfi",
                "fg3_cmf",
                "fg3_volume_price_trend",
                "fg3_price_above_ema50",
                "fg3_ema_alignment",
                "fg3_trend_consistency",
                "fg3_keltner_position",
                "fg3_ema_cross_9_21",
                "fg3_ema_cross_21_50",
                "fg3_macd_cross",
                "fg3_rsi_divergence",
                "fg3_stoch_divergence",
                "fg3_momentum_divergence",
                "fg3_momentum_quality"
            });

            // Williams %R (bounded -100 to 0)
            AddFeatures(map, NormalizationType.None, new[]
            {
                "fg3_williams_r"  // Already bounded
            });

            // Ultimate Oscillator (bounded 0-100)
            AddFeatures(map, NormalizationType.None, new[]
            {
                "fg3_ultimate_oscillator"
            });

            // Labels and targets (keep raw for interpretability)
            AddFeatures(map, NormalizationType.None, new[]
            {
                "label",
                "confidence",
                "long_profit_pips",
                "short_profit_pips",
                "timestamp"  // Keep timestamp as-is
            });

            // ========================================
            // CATEGORY 2: ROBUST SCALING
            // Price and volatility features prone to outliers
            // ========================================

            // EMAs and SMAs (price-based, sensitive to outliers)
            AddFeatures(map, NormalizationType.Robust, new[]
            {
                // M1 moving averages
                "fg1_ema_5",
                "fg1_ema_8",
                
                // M5 moving averages
                "fg3_ema_9",
                "fg3_ema_21",
                "fg3_ema_50",
                "fg3_sma_20",
                "fg3_sma_50",
                
                // VWAP
                "fg2_of_vwap"
            });

            // Bollinger Bands (price levels)
            AddFeatures(map, NormalizationType.Robust, new[]
            {
                "fg1_bb_upper",
                "fg1_bb_lower",
                "fg3_bb_upper",
                "fg3_bb_lower"
            });

            // Channel indicators (price levels)
            AddFeatures(map, NormalizationType.Robust, new[]
            {
                "fg3_keltner_upper",
                "fg3_keltner_lower",
                "fg3_donchian_upper",
                "fg3_donchian_lower"
            });

            // Microstructure price features
            AddFeatures(map, NormalizationType.Robust, new[]
            {
                "fg2_of_micro_price"
            });

            // Volatility features (prone to spikes)
            AddFeatures(map, NormalizationType.Robust, new[]
            {
                // M1 volatility
                "fg1_tr_current",
                "fg1_atr_10",
                "fg1_atr_14",
                "fg1_rv_20",  // Realized volatility
                
                // Advanced volatility
                "fg2_garch_volatility",
                "fg2_vol_of_vol",
                
                // M5 volatility
                "fg3_atr_14",
                "fg3_atr_21",
                "fg3_normalized_atr",
                "fg3_hist_vol_10",
                "fg3_hist_vol_20",
                "fg3_true_range"
            });

            // ========================================
            // CATEGORY 3: STANDARD SCALING (Z-SCORE)
            // Features that are approximately normally distributed
            // ========================================

            // Rate of change and momentum (pip-based, can be positive or negative)
            AddFeatures(map, NormalizationType.Standard, new[]
            {
                // M1 momentum
                "fg1_roc_5",
                "fg1_roc_10",
                "fg1_price_acceleration",
                
                // M5 momentum
                "fg3_momentum_10",
                "fg3_roc_10",
                "fg3_momentum_acceleration",
                "fg3_price_slope",
                "fg3_slope_acceleration",
                "fg3_ema9_slope",
                "fg3_ema21_slope"
            });

            // Spread features (already scaled by 10000 but need standardization)
            AddFeatures(map, NormalizationType.Standard, new[]
            {
                "fg2_of_effective_spread",
                "fg2_of_realized_spread",
                "fg2_of_price_impact",
                "fg2_liquidity_effective_tick",
                "fg2_liquidity_price_impact",
                "fg2_liquidity_tightness",
                "fg2_liquidity_price_efficiency",
                "fg2_liquidity_price_dispersion"
            });

            // VWAP deviations
            AddFeatures(map, NormalizationType.Standard, new[]
            {
                "fg2_of_vwap_deviation",
                "fg2_of_vwap_slope"
            });

            // Unbounded oscillators (can exceed typical ranges)
            AddFeatures(map, NormalizationType.Standard, new[]
            {
                // CCI (Commodity Channel Index, typically -200 to +200 but unbounded)
                "fg1_cci_20",
                
                // MACD (unbounded difference between EMAs)
                "fg1_macd_line",
                "fg1_macd_signal",
                "fg1_macd_histogram",
                "fg3_macd_line",
                "fg3_macd_signal",
                "fg3_macd_histogram"
            });

            // Flow dynamics
            AddFeatures(map, NormalizationType.Standard, new[]
            {
                "fg2_of_flow_momentum",
                "fg2_of_flow_acceleration"
            });

            // Market microstructure metrics
            AddFeatures(map, NormalizationType.Standard, new[]
            {
                "fg1_entropy_20",  // Shannon entropy
                "fg2_fractal_dimension",
                "fg2_detrended_fluctuation",
                "fg3_atr_expansion",
                "fg3_atr_ratio",
                "fg3_vol_ratio",
                "fg3_chaikin_volatility",
                "fg3_volatility_trend"
            });

            // Bollinger Band width (needs standardization)
            AddFeatures(map, NormalizationType.Standard, new[]
            {
                "fg1_bb_width",
                "fg3_bb_width"
            });

            // Trend and distance metrics
            AddFeatures(map, NormalizationType.Standard, new[]
            {
                "fg3_ma_distance",
                "fg3_trend_strength",
                "fg3_donchian_width"
            });

            // ========================================
            // CATEGORY 4: LOG-ROBUST SCALING
            // Volume features (right-skewed with outliers)
            // ========================================

            // Volume and order flow (heavily skewed)
            AddFeatures(map, NormalizationType.LogRobust, new[]
            {
                // Basic volume
                "fg3_volume",
                "fg3_volume_ma",
                
                // Order flow
                "fg2_of_net_flow",
                "fg2_of_cumulative_delta",
                "fg2_of_large_order_ratio",
                
                // On-balance volume
                "fg3_obv",
                "fg3_obv_ma",
                
                // Volume indicators
                "fg3_volume_force",
                "fg3_accumulation_distribution",
                "fg3_vpt",  // Volume Price Trend
                "fg3_emv",  // Ease of Movement
                "fg3_volume_oscillator",
                "fg3_volume_trend",
                
                // Liquidity volume features
                "fg2_liquidity_volume_profile",
                "fg2_liquidity_volume_concentration",
                "fg2_liquidity_volume_dispersion",
                "fg2_liquidity_relative_volume",
                "fg2_liquidity_depth_proxy"
            });

            // Tick-based features (count data, often skewed)
            AddFeatures(map, NormalizationType.LogRobust, new[]
            {
                "fg2_liquidity_tick_rate",
                "fg2_liquidity_tick_acceleration",
                "fg2_liquidity_tick_volatility",
                "fg2_of_trade_intensity"
            });

            // Duration and count features
            AddFeatures(map, NormalizationType.LogRobust, new[]
            {
                "fg2_regime_duration"  // Can have long tails
            });

            // ========================================
            // CATEGORY 5: LOG-STANDARD SCALING
            // Features that benefit from log transform with standard scaling
            // ========================================

            // Liquidity measures (heavy-tailed distributions)
            AddFeatures(map, NormalizationType.LogStandard, new[]
            {
                "fg2_liquidity_amihud_illiquidity",
                "fg2_liquidity_kyle_lambda",
                "fg2_liquidity_roll_measure",
                "fg2_liquidity_hasbrouck_measure"
            });

            // Tail risk metrics (extreme value distributions)
            AddFeatures(map, NormalizationType.LogStandard, new[]
            {
                "fg2_left_tail_risk",
                "fg2_right_tail_risk",
                "fg2_tail_asymmetry"
            });

            return map;
        }

        /// <summary>
        /// Helper method to add multiple features with the same normalization type
        /// </summary>
        private void AddFeatures(Dictionary<string, NormalizationType> map,
            NormalizationType type, string[] features)
        {
            foreach (var feature in features)
            {
                map[feature] = type;
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Fit the normalizer on training data
        /// This should be called only on training data to prevent data leakage
        /// </summary>
        public void Fit(Dictionary<string, double[]> trainingData)
        {
            if (trainingData == null || trainingData.Count == 0)
                throw new ArgumentException("Training data cannot be null or empty");

            Console.WriteLine($"Fitting normalizers on {trainingData.Count} features...");

            _scalers.Clear();
            int fittedCount = 0;

            foreach (var kvp in trainingData)
            {
                var featureName = kvp.Key;
                var values = kvp.Value;

                // Skip if no values
                if (values == null || values.Length == 0)
                    continue;

                // Get normalization type for this feature
                NormalizationType normType;
                if (!_featureNormalizationMap.TryGetValue(featureName, out normType))
                {
                    // Default normalization based on feature name patterns
                    normType = GetDefaultNormalizationType(featureName);
                }

                // Skip if no normalization needed
                if (normType == NormalizationType.None)
                    continue;

                // Create and fit appropriate scaler
                Scaler scaler = CreateScaler(normType);
                if (scaler != null)
                {
                    try
                    {
                        scaler.Fit(values);
                        _scalers[featureName] = scaler;
                        fittedCount++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Warning: Failed to fit scaler for {featureName}: {ex.Message}");
                    }
                }
            }

            _isFitted = true;
            Console.WriteLine($"Fitted {fittedCount} scalers successfully");
        }

        /// <summary>
        /// Transform a single feature value
        /// Used in production for real-time normalization
        /// </summary>
        public double TransformSingle(string featureName, double value)
        {
            // Check if feature needs normalization
            if (!_scalers.ContainsKey(featureName))
            {
                // Either doesn't need normalization or wasn't in training data
                return value;
            }

            try
            {
                return _scalers[featureName].Transform(value);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to transform {featureName}: {ex.Message}");
                return value;  // Return original value on error
            }
        }

        /// <summary>
        /// Transform a complete feature vector
        /// Returns a new dictionary with normalized values
        /// </summary>
        public Dictionary<string, double> Transform(Dictionary<string, double> features)
        {
            if (!_isFitted)
                throw new InvalidOperationException("Normalizer must be fitted before transforming");

            var normalized = new Dictionary<string, double>();

            foreach (var kvp in features)
            {
                normalized[kvp.Key] = TransformSingle(kvp.Key, kvp.Value);
            }

            return normalized;
        }

        /// <summary>
        /// Transform batch of feature vectors
        /// Used for batch processing in training
        /// </summary>
        public Dictionary<string, double[]> TransformBatch(Dictionary<string, double[]> features)
        {
            if (!_isFitted)
                throw new InvalidOperationException("Normalizer must be fitted before transforming");

            var normalized = new Dictionary<string, double[]>();

            foreach (var kvp in features)
            {
                var featureName = kvp.Key;
                var values = kvp.Value;

                if (_scalers.ContainsKey(featureName))
                {
                    // Transform each value
                    var transformedValues = new double[values.Length];
                    for (int i = 0; i < values.Length; i++)
                    {
                        transformedValues[i] = _scalers[featureName].Transform(values[i]);
                    }
                    normalized[featureName] = transformedValues;
                }
                else
                {
                    // No transformation needed
                    normalized[featureName] = values;
                }
            }

            return normalized;
        }

        /// <summary>
        /// Inverse transform for interpretability
        /// Converts normalized values back to original scale
        /// </summary>
        public double InverseTransform(string featureName, double normalizedValue)
        {
            if (_scalers.ContainsKey(featureName))
            {
                return _scalers[featureName].InverseTransform(normalizedValue);
            }
            return normalizedValue;
        }

        #endregion

        #region Persistence

        /// <summary>
        /// Save fitted normalizer to file for production use
        /// </summary>
        public void SaveToFile(string filepath)
        {
            if (!_isFitted)
                throw new InvalidOperationException("Cannot save unfitted normalizer");

            var normalizerData = new NormalizerData
            {
                Version = "1.0",
                CreatedAt = DateTime.UtcNow,
                FeatureScalers = new Dictionary<string, ScalerData>()
            };

            foreach (var kvp in _scalers)
            {
                var scalerData = new ScalerData
                {
                    ScalerType = kvp.Value.GetType().Name,
                    Parameters = kvp.Value.GetParameters()
                };
                normalizerData.FeatureScalers[kvp.Key] = scalerData;
            }

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };

            var json = JsonSerializer.Serialize(normalizerData, options);
            File.WriteAllText(filepath, json);

            Console.WriteLine($"Normalizer saved to {filepath}");
        }

        /// <summary>
        /// Load fitted normalizer from file
        /// </summary>
        public void LoadFromFile(string filepath)
        {
            if (!File.Exists(filepath))
                throw new FileNotFoundException($"Normalizer file not found: {filepath}");

            var json = File.ReadAllText(filepath);
            var normalizerData = JsonSerializer.Deserialize<NormalizerData>(json);

            _scalers.Clear();

            foreach (var kvp in normalizerData.FeatureScalers)
            {
                var featureName = kvp.Key;
                var scalerData = kvp.Value;

                // Create appropriate scaler based on type
                Scaler scaler = scalerData.ScalerType switch
                {
                    nameof(RobustScaler) => new RobustScaler(),
                    nameof(StandardScaler) => new StandardScaler(),
                    nameof(MinMaxScaler) => new MinMaxScaler(),
                    nameof(LogRobustScaler) => new LogRobustScaler(),
                    nameof(LogStandardScaler) => new LogStandardScaler(),
                    _ => throw new NotSupportedException($"Unknown scaler type: {scalerData.ScalerType}")
                };

                scaler.SetParameters(scalerData.Parameters);
                _scalers[featureName] = scaler;
            }

            _isFitted = true;
            Console.WriteLine($"Loaded {_scalers.Count} scalers from {filepath}");
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Create a scaler instance based on normalization type
        /// </summary>
        private Scaler CreateScaler(NormalizationType type)
        {
            return type switch
            {
                NormalizationType.Robust => new RobustScaler(),
                NormalizationType.Standard => new StandardScaler(),
                NormalizationType.MinMax => new MinMaxScaler(),
                NormalizationType.LogRobust => new LogRobustScaler(),
                NormalizationType.LogStandard => new LogStandardScaler(),
                _ => null
            };
        }

        /// <summary>
        /// Get default normalization type based on feature name patterns
        /// Used for features not explicitly mapped
        /// </summary>
        private NormalizationType GetDefaultNormalizationType(string featureName)
        {
            // Check for patterns that indicate no normalization
            string[] noNormPatterns = { "_pct", "_ratio", "pattern_", "_prob",
                                       "_confidence", "_type", "_divergence" };
            if (noNormPatterns.Any(pattern => featureName.Contains(pattern)))
                return NormalizationType.None;

            // Check for RSI, Stochastic, Aroon (bounded indicators)
            if (featureName.Contains("_rsi") || featureName.Contains("_stoch") ||
                featureName.Contains("_aroon"))
                return NormalizationType.None;

            // Check for volume features
            if (featureName.Contains("volume") || featureName.Contains("_obv") ||
                featureName.Contains("_flow"))
                return NormalizationType.LogRobust;

            // Check for price features
            if (featureName.Contains("_ema") || featureName.Contains("_sma") ||
                featureName.Contains("_price"))
                return NormalizationType.Robust;

            // Check for volatility features
            if (featureName.Contains("_atr") || featureName.Contains("_vol") ||
                featureName.Contains("_tr"))
                return NormalizationType.Robust;

            // Default to standard scaling
            return NormalizationType.Standard;
        }

        /// <summary>
        /// Validate that features are properly normalized
        /// Useful for debugging and quality checks
        /// </summary>
        public Dictionary<string, NormalizationStats> ValidateNormalization(
            Dictionary<string, double[]> normalizedData)
        {
            var stats = new Dictionary<string, NormalizationStats>();

            foreach (var kvp in normalizedData)
            {
                var featureName = kvp.Key;
                var values = kvp.Value;

                if (values == null || values.Length == 0)
                    continue;

                var featureStats = new NormalizationStats
                {
                    FeatureName = featureName,
                    Mean = values.Average(),
                    StdDev = CalculateStdDev(values),
                    Min = values.Min(),
                    Max = values.Max(),
                    HasNaN = values.Any(double.IsNaN),
                    HasInfinity = values.Any(double.IsInfinity)
                };

                // Check if normalization is as expected
                if (_scalers.ContainsKey(featureName))
                {
                    var scalerType = _scalers[featureName].GetType().Name;
                    featureStats.ScalerType = scalerType;

                    // Validate based on scaler type
                    switch (scalerType)
                    {
                        case nameof(StandardScaler):
                        case nameof(LogStandardScaler):
                            // Should have mean ≈ 0, std ≈ 1
                            featureStats.IsValid = Math.Abs(featureStats.Mean) < 0.1 &&
                                                  Math.Abs(featureStats.StdDev - 1) < 0.1;
                            break;

                        case nameof(MinMaxScaler):
                            // Should be in range [0, 1]
                            featureStats.IsValid = featureStats.Min >= -0.01 &&
                                                  featureStats.Max <= 1.01;
                            break;

                        case nameof(RobustScaler):
                        case nameof(LogRobustScaler):
                            // Median should be ≈ 0
                            var median = GetMedian(values);
                            featureStats.IsValid = Math.Abs(median) < 0.1;
                            break;

                        default:
                            featureStats.IsValid = true;
                            break;
                    }
                }
                else
                {
                    featureStats.ScalerType = "None";
                    featureStats.IsValid = !featureStats.HasNaN && !featureStats.HasInfinity;
                }

                stats[featureName] = featureStats;
            }

            return stats;
        }

        private double CalculateStdDev(double[] values)
        {
            if (values.Length < 2) return 0;
            var mean = values.Average();
            var sumSquares = values.Sum(v => Math.Pow(v - mean, 2));
            return Math.Sqrt(sumSquares / (values.Length - 1));
        }

        private double GetMedian(double[] values)
        {
            var sorted = values.OrderBy(v => v).ToArray();
            int n = sorted.Length;
            return n % 2 == 0
                ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                : sorted[n / 2];
        }

        #endregion

        #region Data Classes

        /// <summary>
        /// Data class for persisting normalizer
        /// </summary>
        public class NormalizerData
        {
            public string Version { get; set; }
            public DateTime CreatedAt { get; set; }
            public Dictionary<string, ScalerData> FeatureScalers { get; set; }
        }

        /// <summary>
        /// Data class for persisting individual scalers
        /// </summary>
        public class ScalerData
        {
            public string ScalerType { get; set; }
            public Dictionary<string, double> Parameters { get; set; }
        }

        /// <summary>
        /// Statistics for validation
        /// </summary>
        public class NormalizationStats
        {
            public string FeatureName { get; set; }
            public string ScalerType { get; set; }
            public double Mean { get; set; }
            public double StdDev { get; set; }
            public double Min { get; set; }
            public double Max { get; set; }
            public bool HasNaN { get; set; }
            public bool HasInfinity { get; set; }
            public bool IsValid { get; set; }
        }

        #endregion
    }
}