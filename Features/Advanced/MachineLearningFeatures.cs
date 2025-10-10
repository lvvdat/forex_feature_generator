using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Core;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Advanced
{
    /// <summary>
    /// Machine Learning optimized features including feature interactions,
    /// polynomial features, normalized representations, and ML-ready transformations
    /// </summary>
    public class MachineLearningFeatures : BaseFeatureCalculator
    {
        public override string Name => "MachineLearning";
        public override string Category => "ML_Optimized";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 7;

        private readonly RollingWindow<double> _priceHistory = new(100);
        private readonly RollingWindow<double> _volumeHistory = new(100);
        private readonly RollingWindow<double> _volatilityHistory = new(100);
        private readonly RollingWindow<MLSnapshot> _mlHistory = new(50);

        // Rolling statistics for normalization
        private double _priceRollingMean = 0;
        private double _priceRollingStd = 0;
        private double _volumeRollingMean = 0;
        private double _volumeRollingStd = 0;

        public class MLSnapshot
        {
            public DateTime Timestamp { get; set; }
            public double NormalizedPrice { get; set; }
            public double NormalizedVolume { get; set; }
            public double TrendScore { get; set; }
            public double MomentumScore { get; set; }
        }

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // Update rolling statistics
            UpdateRollingStatistics(bars, currentIndex);

            // ===== NORMALIZED FEATURES (Z-SCORE) =====
            var normalizedPrice = SafeDiv(close - _priceRollingMean, _priceRollingStd);
            output.AddFeature("07_ml_price_zscore", normalizedPrice);

            var normalizedVolume = SafeDiv(bar.TickVolume - _volumeRollingMean, _volumeRollingStd);
            output.AddFeature("07_ml_volume_zscore", normalizedVolume);

            // ===== FEATURE INTERACTIONS =====
            // Price-Volume interaction
            var priceVolumeInteraction = normalizedPrice * normalizedVolume;
            output.AddFeature("07_ml_price_volume_interaction", priceVolumeInteraction);

            // Trend-Momentum interaction
            if (currentIndex >= 50)
            {
                var trendStrength = CalculateTrendStrength(bars, currentIndex);
                var momentumStrength = CalculateMomentumStrength(bars, currentIndex);
                output.AddFeature("07_ml_trend_momentum_interaction", trendStrength * momentumStrength);
            }

            // Volatility-Volume interaction
            var volatility = CalculateATR(bars, currentIndex, 14);
            var volVolInteraction = SafeDiv(volatility * bar.TickVolume, _volumeRollingMean);
            output.AddFeature("07_ml_volatility_volume_interaction", volVolInteraction);

            // ===== POLYNOMIAL FEATURES =====
            // Quadratic price momentum
            var returns = Math.Log(close / (double)bars[currentIndex - 10].Close);
            output.AddFeature("07_ml_returns_squared", returns * returns);
            output.AddFeature("07_ml_returns_cubed", returns * returns * returns);

            // Volume concentration (Gini coefficient approximation)
            var volumeGini = CalculateVolumeConcentration(bars, currentIndex);
            output.AddFeature("07_ml_volume_gini", volumeGini);

            // ===== RATIO FEATURES =====
            var sma20 = CalculateSMA(bars, currentIndex, 20);
            var ema20 = CalculateEMA(bars, currentIndex, 20);

            // Price to MA ratios
            output.AddFeature("07_ml_price_to_sma20_ratio", SafeDiv(close, sma20));
            output.AddFeature("07_ml_price_to_ema20_ratio", SafeDiv(close, ema20));

            // EMA/SMA ratio (trend quality indicator)
            output.AddFeature("07_ml_ema_sma_ratio", SafeDiv(ema20, sma20));

            // Volume ratio to various averages
            if (_volumeHistory.Count >= 20)
            {
                var vol5 = _volumeHistory.GetValues().Take(5).Average();
                var vol20 = _volumeHistory.GetValues().Take(20).Average();
                output.AddFeature("07_ml_volume_short_long_ratio", SafeDiv(vol5, vol20));
            }
            else
            {
                output.AddFeature("07_ml_volume_short_long_ratio", 0);
            }

            // ===== ROLLING STATISTICS =====
            if (currentIndex >= 30)
            {
                // Rolling correlation (price vs volume)
                var priceVolCorr = CalculateRollingCorrelation(
                    bars.Skip(currentIndex - 19).Take(20).Select(b => (double)b.Close).ToArray(),
                    bars.Skip(currentIndex - 19).Take(20).Select(b => (double)b.TickVolume).ToArray()
                );
                output.AddFeature("07_ml_price_volume_correlation", priceVolCorr);

                // Rolling covariance
                var priceVolCov = CalculateRollingCovariance(
                    bars.Skip(currentIndex - 19).Take(20).Select(b => (double)b.Close).ToArray(),
                    bars.Skip(currentIndex - 19).Take(20).Select(b => (double)b.TickVolume).ToArray()
                );
                output.AddFeature("07_ml_price_volume_covariance", priceVolCov);
            }

            // ===== COMPOSITE SCORES =====
            // Trend Score (composite of multiple trend indicators)
            var trendScore = CalculateCompositeTrendScore(bars, currentIndex);
            output.AddFeature("07_ml_composite_trend_score", trendScore);

            // Momentum Score (composite of multiple momentum indicators)
            var momentumScore = CalculateCompositeMomentumScore(bars, currentIndex);
            output.AddFeature("07_ml_composite_momentum_score", momentumScore);

            // Volatility Score (composite of volatility measures)
            var volatilityScore = CalculateCompositeVolatilityScore(bars, currentIndex);
            output.AddFeature("07_ml_composite_volatility_score", volatilityScore);

            // Quality Score (overall market quality for trading)
            var qualityScore = CalculateMarketQualityScore(bars, currentIndex);
            output.AddFeature("07_ml_market_quality_score", qualityScore);

            // ===== PERCENTILE RANKS =====
            if (_priceHistory.Count >= 50)
            {
                var pricePercentile = CalculatePercentileRank(_priceHistory.GetValues().Take(50).ToArray(), close);
                output.AddFeature("07_ml_price_percentile_50", pricePercentile);
            }
            else
            {
                output.AddFeature("07_ml_price_percentile_50", 0.5);
            }

            if (_volumeHistory.Count >= 50)
            {
                var volumePercentile = CalculatePercentileRank(_volumeHistory.GetValues().Take(50).ToArray(), bar.TickVolume);
                output.AddFeature("07_ml_volume_percentile_50", volumePercentile);
            }
            else
            {
                output.AddFeature("07_ml_volume_percentile_50", 0.5);
            }

            // ===== DISTANCE METRICS =====
            if (currentIndex >= 50)
            {
                // Euclidean distance from recent mean state
                var distanceFromMean = CalculateStateDistance(bars, currentIndex);
                output.AddFeature("07_ml_state_distance_from_mean", distanceFromMean);

                // Mahalanobis-like distance (scaled by volatility)
                var scaledDistance = SafeDiv(distanceFromMean, CalculateATR(bars, currentIndex, 14));
                output.AddFeature("07_ml_scaled_state_distance", scaledDistance);
            }

            // ===== ENTROPY MEASURES =====
            if (currentIndex >= 30)
            {
                // Price entropy
                var priceEntropy = CalculateLocalEntropy(
                    bars.Skip(currentIndex - 29).Take(30).Select(b => (double)b.Close).ToArray()
                );
                output.AddFeature("07_ml_price_entropy", priceEntropy);

                // Volume entropy
                var volumeEntropy = CalculateLocalEntropy(
                    bars.Skip(currentIndex - 29).Take(30).Select(b => (double)b.TickVolume).ToArray()
                );
                output.AddFeature("07_ml_volume_entropy", volumeEntropy);
            }

            // ===== FEATURE ENGINEERING FOR GRADIENT BOOSTING =====
            // Binned features (categorical-like for tree-based models)
            output.AddFeature("07_ml_price_bin", BinValue(close, _priceRollingMean, _priceRollingStd));
            output.AddFeature("07_ml_volume_bin", BinValue(bar.TickVolume, _volumeRollingMean, _volumeRollingStd));

            // One-hot encoded hour of day (for time patterns)
            var hour = bar.Timestamp.Hour;
            output.AddFeature("07_ml_hour_asian", (hour >= 0 && hour < 8) ? 1.0 : 0.0);
            output.AddFeature("07_ml_hour_european", (hour >= 8 && hour < 16) ? 1.0 : 0.0);
            output.AddFeature("07_ml_hour_american", (hour >= 16 && hour < 24) ? 1.0 : 0.0);

            // ===== LAGGED FEATURES =====
            output.AddFeature("07_ml_price_lag_1", (double)bars[currentIndex - 1].Close);
            output.AddFeature("07_ml_price_lag_3", (double)bars[currentIndex - 3].Close);
            output.AddFeature("07_ml_price_lag_5", (double)bars[currentIndex - 5].Close);

            // Percentage changes
            output.AddFeature("07_ml_pct_change_lag_1", SafeDiv(close - (double)bars[currentIndex - 1].Close, (double)bars[currentIndex - 1].Close) * 100);
            output.AddFeature("07_ml_pct_change_lag_5", SafeDiv(close - (double)bars[currentIndex - 5].Close, (double)bars[currentIndex - 5].Close) * 100);

            // ===== TECHNICAL INDICATOR COMBINATIONS =====
            if (currentIndex >= 50)
            {
                // RSI-Stochastic combination
                var rsi = CalculateRSI(bars, currentIndex, 14);
                var stoch = CalculateStochastic(bars, currentIndex, 14);
                output.AddFeature("07_ml_rsi_stoch_avg", (rsi + stoch) / 2);
                output.AddFeature("07_ml_rsi_stoch_diff", Math.Abs(rsi - stoch));

                // MACD-ADX combination
                var macd = CalculateMACD(bars, currentIndex);
                var adx = CalculateADX(bars, currentIndex, 14);
                output.AddFeature("07_ml_macd_adx_product", macd * adx);
            }

            // Update histories
            _priceHistory.Add(close);
            _volumeHistory.Add(bar.TickVolume);
            _volatilityHistory.Add(CalculateATR(bars, currentIndex, 14));

            _mlHistory.Add(new MLSnapshot
            {
                Timestamp = bar.Timestamp,
                NormalizedPrice = normalizedPrice,
                NormalizedVolume = normalizedVolume,
                TrendScore = trendScore,
                MomentumScore = momentumScore
            });
        }

        private void UpdateRollingStatistics(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return;

            var prices = new List<double>();
            var volumes = new List<double>();

            for (int i = currentIndex - 49; i <= currentIndex; i++)
            {
                prices.Add((double)bars[i].Close);
                volumes.Add(bars[i].TickVolume);
            }

            _priceRollingMean = prices.Average();
            _priceRollingStd = Math.Sqrt(prices.Select(p => Math.Pow(p - _priceRollingMean, 2)).Average());

            _volumeRollingMean = volumes.Average();
            _volumeRollingStd = Math.Sqrt(volumes.Select(v => Math.Pow(v - _volumeRollingMean, 2)).Average());
        }

        private double CalculateTrendStrength(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var ema9 = CalculateEMA(bars, currentIndex, 9);
            var ema21 = CalculateEMA(bars, currentIndex, 21);
            var ema50 = CalculateEMA(bars, currentIndex, 50);

            var strength = 0.0;
            if (ema9 > ema21 && ema21 > ema50) strength = 1.0;
            else if (ema9 < ema21 && ema21 < ema50) strength = -1.0;
            else strength = SafeDiv(ema9 - ema50, ema50);

            return strength;
        }

        private double CalculateMomentumStrength(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var rsi = CalculateRSI(bars, currentIndex, 14);
            var roc = SafeDiv((double)(bars[currentIndex].Close - bars[currentIndex - 10].Close), (double)bars[currentIndex - 10].Close);

            // Normalize to [-1, 1]
            var rsiNorm = (rsi - 50) / 50;
            var rocNorm = Math.Tanh(roc * 100);

            return (rsiNorm + rocNorm) / 2;
        }

        private double CalculateVolumeConcentration(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var volumes = new List<double>();
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                volumes.Add(bars[i].TickVolume);
            }

            volumes.Sort();
            var total = volumes.Sum();
            if (total < 1e-10) return 0;

            double gini = 0;
            for (int i = 0; i < volumes.Count; i++)
            {
                gini += (2 * (i + 1) - volumes.Count - 1) * volumes[i];
            }

            return gini / (volumes.Count * total);
        }

        private double CalculateRollingCorrelation(double[] x, double[] y)
        {
            if (x.Length != y.Length || x.Length < 2) return 0;

            var n = x.Length;
            var sumX = x.Sum();
            var sumY = y.Sum();
            var sumXY = x.Zip(y, (a, b) => a * b).Sum();
            var sumX2 = x.Sum(a => a * a);
            var sumY2 = y.Sum(b => b * b);

            var numerator = n * sumXY - sumX * sumY;
            var denominator = Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

            return denominator > 1e-10 ? numerator / denominator : 0;
        }

        private double CalculateRollingCovariance(double[] x, double[] y)
        {
            if (x.Length != y.Length || x.Length < 2) return 0;

            var meanX = x.Average();
            var meanY = y.Average();

            return x.Zip(y, (a, b) => (a - meanX) * (b - meanY)).Average();
        }

        private double CalculateCompositeTrendScore(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return 0;

            var scores = new List<double>();

            // ADX contribution
            var adx = CalculateADX(bars, currentIndex, 14);
            scores.Add(Math.Min(1.0, adx / 50));

            // Linear regression slope
            var slope = CalculateLinearSlope(bars, currentIndex, 20);
            scores.Add(Math.Tanh(slope * 1000));

            // EMA alignment
            var ema9 = CalculateEMA(bars, currentIndex, 9);
            var ema21 = CalculateEMA(bars, currentIndex, 21);
            var ema50 = CalculateEMA(bars, currentIndex, 50);

            var alignment = 0.0;
            if (ema9 > ema21 && ema21 > ema50) alignment = 1.0;
            else if (ema9 < ema21 && ema21 < ema50) alignment = -1.0;
            scores.Add(alignment);

            return scores.Average();
        }

        private double CalculateCompositeMomentumScore(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 26) return 0;

            var scores = new List<double>();

            // RSI
            var rsi = CalculateRSI(bars, currentIndex, 14);
            scores.Add((rsi - 50) / 50);

            // MACD
            var macd = CalculateMACD(bars, currentIndex);
            scores.Add(Math.Tanh(macd * 1000));

            // ROC
            var roc = SafeDiv((double)(bars[currentIndex].Close - bars[currentIndex - 10].Close), (double)bars[currentIndex - 10].Close);
            scores.Add(Math.Tanh(roc * 100));

            return scores.Average();
        }

        private double CalculateCompositeVolatilityScore(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return 0;

            var atr = CalculateATR(bars, currentIndex, 14);
            var close = (double)bars[currentIndex].Close;
            var atrPct = SafeDiv(atr, close);

            var stdDev = CalculateStdDev(bars, currentIndex, 20);
            var stdPct = SafeDiv(stdDev, close);

            // Normalize to [0, 1]
            return (Math.Min(1.0, atrPct * 1000) + Math.Min(1.0, stdPct * 1000)) / 2;
        }

        private double CalculateMarketQualityScore(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return 0.5;

            var scores = new List<double>();

            // Volume consistency
            var volumeStd = _volumeHistory.Count >= 20 ?
                Math.Sqrt(_volumeHistory.GetValues().Take(20).Select(v => Math.Pow(v - _volumeHistory.GetValues().Take(20).Average(), 2)).Average()) : 0;
            var volumeConsistency = 1.0 - Math.Min(1.0, SafeDiv(volumeStd, _volumeRollingMean));
            scores.Add(volumeConsistency);

            // Spread quality (lower is better)
            var avgSpread = (double)bars[currentIndex].AvgSpread;
            var spreadQuality = 1.0 - Math.Min(1.0, avgSpread * 10000);
            scores.Add(spreadQuality);

            // Trend clarity
            var adx = CalculateADX(bars, currentIndex, 14);
            var trendClarity = Math.Min(1.0, adx / 50);
            scores.Add(trendClarity);

            return scores.Average();
        }

        private double CalculatePercentileRank(double[] values, double target)
        {
            if (values.Length == 0) return 0.5;

            var sorted = values.OrderBy(v => v).ToArray();
            var count = sorted.Count(v => v < target);

            return (double)count / values.Length;
        }

        private double CalculateStateDistance(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Calculate Euclidean distance from mean state
            var recentPrices = new List<double>();
            var recentVolumes = new List<double>();

            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                recentPrices.Add((double)bars[i].Close);
                recentVolumes.Add(bars[i].TickVolume);
            }

            var priceMean = recentPrices.Average();
            var volumeMean = recentVolumes.Average();

            var currentPrice = (double)bars[currentIndex].Close;
            var currentVolume = (double)bars[currentIndex].TickVolume;

            var priceDistance = Math.Pow(currentPrice - priceMean, 2);
            var volumeDistance = Math.Pow(currentVolume - volumeMean, 2);

            return Math.Sqrt(priceDistance + volumeDistance);
        }

        private double CalculateLocalEntropy(double[] values)
        {
            if (values.Length == 0) return 0;

            // Discretize into bins
            var bins = 10;
            var min = values.Min();
            var max = values.Max();
            var binWidth = (max - min) / bins;

            if (binWidth < 1e-10) return 0;

            var counts = new int[bins];
            foreach (var value in values)
            {
                var bin = (int)((value - min) / binWidth);
                if (bin >= bins) bin = bins - 1;
                if (bin < 0) bin = 0;
                counts[bin]++;
            }

            double entropy = 0;
            foreach (var count in counts)
            {
                if (count > 0)
                {
                    var p = (double)count / values.Length;
                    entropy -= p * Math.Log(p, 2);
                }
            }

            return entropy;
        }

        private double BinValue(double value, double mean, double std)
        {
            if (std < 1e-10) return 0;

            var zScore = (value - mean) / std;

            if (zScore < -2) return -3;
            if (zScore < -1) return -2;
            if (zScore < -0.5) return -1;
            if (zScore < 0.5) return 0;
            if (zScore < 1) return 1;
            if (zScore < 2) return 2;
            return 3;
        }

        private double CalculateRSI(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 50;

            double gains = 0;
            double losses = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var change = (double)(bars[i].Close - bars[i - 1].Close);
                if (change > 0) gains += change;
                else losses += Math.Abs(change);
            }

            var avgGain = gains / period;
            var avgLoss = losses / period;

            if (avgLoss < 1e-10) return 100;

            var rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }

        private double CalculateStochastic(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 50;

            double high = double.MinValue;
            double low = double.MaxValue;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                high = Math.Max(high, (double)bars[i].High);
                low = Math.Min(low, (double)bars[i].Low);
            }

            var close = (double)bars[currentIndex].Close;
            return SafeDiv(close - low, high - low) * 100;
        }

        private double CalculateMACD(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 26) return 0;

            var ema12 = CalculateEMA(bars, currentIndex, 12);
            var ema26 = CalculateEMA(bars, currentIndex, 26);

            return ema12 - ema26;
        }

        private double CalculateADX(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period * 2) return 0;

            var dmPlus = new List<double>();
            var dmMinus = new List<double>();
            var tr = new List<double>();

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var highDiff = (double)(bars[i].High - bars[i - 1].High);
                var lowDiff = (double)(bars[i - 1].Low - bars[i].Low);

                dmPlus.Add(highDiff > lowDiff && highDiff > 0 ? highDiff : 0);
                dmMinus.Add(lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0);
                tr.Add(CalculateTrueRange(bars, i));
            }

            var avgDmPlus = dmPlus.Average();
            var avgDmMinus = dmMinus.Average();
            var avgTr = tr.Average();

            if (avgTr < 1e-10) return 0;

            var diPlus = 100 * avgDmPlus / avgTr;
            var diMinus = 100 * avgDmMinus / avgTr;

            var dx = Math.Abs(diPlus - diMinus) / (diPlus + diMinus) * 100;

            return dx;
        }

        private double CalculateLinearSlope(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            var x = Enumerable.Range(0, period).Select(i => (double)i).ToArray();
            var y = new double[period];

            for (int i = 0; i < period; i++)
            {
                y[i] = (double)bars[currentIndex - period + 1 + i].Close;
            }

            var n = period;
            var sumX = x.Sum();
            var sumY = y.Sum();
            var sumXY = x.Zip(y, (a, b) => a * b).Sum();
            var sumX2 = x.Sum(a => a * a);

            return SafeDiv(n * sumXY - sumX * sumY, n * sumX2 - sumX * sumX);
        }

        public override void Reset()
        {
            _priceHistory.Clear();
            _volumeHistory.Clear();
            _volatilityHistory.Clear();
            _mlHistory.Clear();
        }
    }
}