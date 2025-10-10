using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Core directional features optimized for 3-class prediction
    /// These are the primary features that directly indicate direction
    /// </summary>
    public class DirectionalFeatures : BaseFeatureCalculator
    {
        public override string Name => "Directional";
        public override string Category => "Primary";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 1;  // Highest priority

        private readonly RollingWindow<double> _priceHistory = new(100);
        private readonly RollingWindow<double> _momentumHistory = new(50);
        private readonly RollingWindow<double> _volumeHistory = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return;  // Need sufficient history

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            _priceHistory.Add(close);

            // ===== 1. MOMENTUM-BASED DIRECTIONAL FEATURES =====
            // These directly correlate with label direction

            // Short-term momentum (5-bar) - normalized
            var momentum5 = close - (double)bars[currentIndex - 5].Close;
            var momentumZ5 = CalculateMomentumZScore(bars, currentIndex, 5);
            output.AddFeature("dir_momentum_z5", momentumZ5);
            output.AddFeature("dir_momentum_signal_5", CreateDirectionalSignal(momentumZ5, 1.5, -1.5));

            // Medium-term momentum (10-bar)
            var momentumZ10 = CalculateMomentumZScore(bars, currentIndex, 10);
            output.AddFeature("dir_momentum_z10", momentumZ10);
            output.AddFeature("dir_momentum_signal_10", CreateDirectionalSignal(momentumZ10, 1.0, -1.0));

            // Momentum acceleration (2nd derivative)
            var momentumAccel = CalculateMomentumAcceleration(bars, currentIndex);
            output.AddFeature("dir_momentum_accel", momentumAccel);
            output.AddFeature("dir_momentum_accel_signal", CreateDirectionalSignal(momentumAccel, 0.5, -0.5));

            // Momentum quality (consistency of direction)
            var momentumQuality = CalculateMomentumQualityScore(bars, currentIndex);
            output.AddFeature("dir_momentum_quality", momentumQuality);

            // ===== 2. PRICE ACTION DIRECTIONAL FEATURES =====
            // Based on candlestick patterns and price structure

            // Directional candle strength
            var candleDirection = CalculateCandleDirection(bar);
            output.AddFeature("dir_candle_direction", candleDirection);

            // Multi-bar directional pattern
            var patternDirection = CalculateMultiBarPattern(bars, currentIndex);
            output.AddFeature("dir_pattern_strength", patternDirection);

            // Higher high/lower low sequence
            var hhllSignal = CalculateHHLLSignal(bars, currentIndex);
            output.AddFeature("dir_hhll_signal", hhllSignal);

            // Price position relative to recent range
            var pricePosition = CalculatePricePosition(bars, currentIndex, 20);
            output.AddFeature("dir_price_position", pricePosition);
            output.AddFeature("dir_price_breakout", Math.Abs(pricePosition) > 0.8 ? Math.Sign(pricePosition) : 0);

            // ===== 3. VOLUME-BASED DIRECTIONAL FEATURES =====
            // Volume confirms direction

            var volumeDirection = CalculateVolumeDirection(bars[currentIndex]);
            output.AddFeature("dir_volume_direction", volumeDirection);

            // Volume-weighted directional pressure
            var volumePressure = CalculateVolumePressure(bars, currentIndex);
            output.AddFeature("dir_volume_pressure", volumePressure);
            output.AddFeature("dir_volume_signal", CreateDirectionalSignal(volumePressure, 0.3, -0.3));

            // Volume momentum correlation
            var volumeMomentumCorr = CalculateVolumeMomentumCorrelation(bars, currentIndex);
            output.AddFeature("dir_vol_mom_correlation", volumeMomentumCorr);

            // ===== 4. TREND STRENGTH FEATURES =====
            // Identifies strong directional moves vs choppy markets

            // ADX-based trend strength
            var trendStrength = CalculateTrendStrength(bars, currentIndex);
            output.AddFeature("dir_trend_strength", trendStrength);

            // Directional movement
            var (dmPlus, dmMinus) = CalculateDirectionalMovement(bars, currentIndex);
            output.AddFeature("dir_dm_plus", dmPlus);
            output.AddFeature("dir_dm_minus", dmMinus);
            output.AddFeature("dir_dm_signal", dmPlus > dmMinus ? 1.0 : dmMinus > dmPlus ? -1.0 : 0.0);

            // Trend efficiency (how directly price moves)
            var efficiency = CalculateTrendEfficiency(bars, currentIndex, 10);
            output.AddFeature("dir_trend_efficiency", efficiency);

            // ===== 5. SUPPORT/RESISTANCE PROXIMITY =====
            // Distance from key levels affects direction probability

            var (supportDist, resistanceDist) = CalculateSRDistance(bars, currentIndex);
            output.AddFeature("dir_support_distance_norm", supportDist);
            output.AddFeature("dir_resistance_distance_norm", resistanceDist);

            // SR bounce probability
            var bounceProbability = CalculateBounceProbability(supportDist, resistanceDist);
            output.AddFeature("dir_bounce_probability", bounceProbability);

            // ===== 6. COMPOSITE DIRECTIONAL SIGNALS =====
            // Combines multiple indicators for robust prediction

            // Primary composite signal
            var primaryComposite = CreateCompositeSignal(
                (momentumZ5, 0.3),
                (momentumZ10, 0.2),
                (volumePressure, 0.2),
                (patternDirection, 0.15),
                (dmPlus - dmMinus, 0.15)
            );
            output.AddFeature("dir_composite_primary", primaryComposite);

            // Confirmation composite (for high confidence trades)
            var confirmationComposite = CreateCompositeSignal(
                (momentumQuality, 0.25),
                (volumeMomentumCorr, 0.25),
                (efficiency, 0.25),
                (trendStrength > 0.3 ? primaryComposite : 0, 0.25)
            );
            output.AddFeature("dir_composite_confirmation", confirmationComposite);

            // Final directional probability
            var directionalProb = CalculateDirectionalProbability(primaryComposite, confirmationComposite, trendStrength);
            output.AddFeature("dir_probability", directionalProb);

            // Signal confidence score
            var confidence = Math.Abs(directionalProb) * trendStrength * momentumQuality;
            output.AddFeature("dir_confidence", confidence);

            // ===== 7. REVERSAL DETECTION FEATURES =====
            // Identifies potential direction changes

            // Momentum divergence
            var divergence = CalculateMomentumDivergence(bars, currentIndex);
            output.AddFeature("dir_divergence", divergence);

            // Exhaustion signal
            var exhaustion = CalculateExhaustionSignal(bars, currentIndex);
            output.AddFeature("dir_exhaustion", exhaustion);

            // Mean reversion probability
            var meanReversionProb = CalculateMeanReversionProbability(bars, currentIndex);
            output.AddFeature("dir_mean_reversion_prob", meanReversionProb);
        }

        // ===== HELPER METHODS =====

        private double CalculateMomentumZScore(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            var momentums = new List<double>();
            for (int i = currentIndex - 30; i <= currentIndex; i++)
            {
                if (i >= period)
                {
                    var mom = (double)(bars[i].Close - bars[i - period].Close);
                    momentums.Add(mom);
                }
            }

            if (momentums.Count < 2) return 0;

            var currentMom = (double)(bars[currentIndex].Close - bars[currentIndex - period].Close);
            var mean = momentums.Average();
            var stdDev = Math.Sqrt(momentums.Select(m => Math.Pow(m - mean, 2)).Average());

            return CalculateZScore(currentMom, mean, stdDev);
        }

        private double CalculateMomentumAcceleration(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 15) return 0;

            var mom1 = (double)(bars[currentIndex].Close - bars[currentIndex - 5].Close);
            var mom2 = (double)(bars[currentIndex - 5].Close - bars[currentIndex - 10].Close);
            var mom3 = (double)(bars[currentIndex - 10].Close - bars[currentIndex - 15].Close);

            var accel1 = mom1 - mom2;
            var accel2 = mom2 - mom3;

            return Sigmoid((accel1 - accel2) * 10000);  // Normalized acceleration
        }

        private double CalculateMomentumQualityScore(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var momentums = new List<double>();
            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                momentums.Add((double)(bars[i].Close - bars[i - 1].Close));
            }

            return CalculateMomentumQuality(momentums);
        }

        private double CalculateCandleDirection(OhlcBar bar)
        {
            var body = (double)(bar.Close - bar.Open);
            var range = (double)(bar.High - bar.Low);

            if (range < 1e-10) return 0;

            var bodyRatio = body / range;
            var upperWick = (double)(bar.High - Math.Max(bar.Open, bar.Close)) / range;
            var lowerWick = (double)(Math.Min(bar.Open, bar.Close) - bar.Low) / range;

            // Strong bullish: large positive body, small upper wick
            if (bodyRatio > 0.6 && upperWick < 0.2) return 1.0;

            // Strong bearish: large negative body, small lower wick
            if (bodyRatio < -0.6 && lowerWick < 0.2) return -1.0;

            // Hammer pattern (bullish reversal)
            if (Math.Abs(bodyRatio) < 0.3 && lowerWick > 0.6) return 0.5;

            // Shooting star (bearish reversal)
            if (Math.Abs(bodyRatio) < 0.3 && upperWick > 0.6) return -0.5;

            return bodyRatio;  // Default to normalized body ratio
        }

        private double CalculateMultiBarPattern(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            double patternScore = 0;
            int weight = 3;  // Recent bars have more weight

            for (int i = currentIndex - 2; i <= currentIndex; i++)
            {
                var direction = CalculateCandleDirection(bars[i]);
                patternScore += direction * weight;
                weight--;
            }

            return Sigmoid(patternScore / 6);  // Normalize to [-1, 1]
        }

        private double CalculateHHLLSignal(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Find swing highs and lows
            var highs = new List<double>();
            var lows = new List<double>();

            for (int i = currentIndex - 20; i <= currentIndex; i += 5)
            {
                var localHigh = double.MinValue;
                var localLow = double.MaxValue;

                for (int j = i - 2; j <= i + 2 && j <= currentIndex; j++)
                {
                    if (j >= 0)
                    {
                        localHigh = Math.Max(localHigh, (double)bars[j].High);
                        localLow = Math.Min(localLow, (double)bars[j].Low);
                    }
                }

                highs.Add(localHigh);
                lows.Add(localLow);
            }

            // Check for higher highs and higher lows (uptrend)
            int hhCount = 0, llCount = 0;
            for (int i = 1; i < highs.Count; i++)
            {
                if (highs[i] > highs[i - 1]) hhCount++;
                if (lows[i] > lows[i - 1]) llCount++;
            }

            if (hhCount > highs.Count / 2 && llCount > lows.Count / 2) return 1.0;
            if (hhCount < highs.Count / 3 && llCount < lows.Count / 3) return -1.0;

            return 0.0;
        }

        private double CalculatePricePosition(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            double highest = double.MinValue;
            double lowest = double.MaxValue;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                highest = Math.Max(highest, (double)bars[i].High);
                lowest = Math.Min(lowest, (double)bars[i].Low);
            }

            var close = (double)bars[currentIndex].Close;
            return NormalizeToRange(close, lowest, highest);
        }

        private double CalculateVolumeDirection(OhlcBar bar)
        {
            var totalVolume = bar.UpVolume + bar.DownVolume;
            if (totalVolume < 1e-10) return 0;

            return ((double)bar.UpVolume - (double)bar.DownVolume) / (double)totalVolume;
        }

        private double CalculateVolumePressure(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            double buyPressure = 0;
            double sellPressure = 0;

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                var priceChange = (double)(bars[i].Close - bars[i].Open);
                var volume = bars[i].TickVolume;

                if (priceChange > 0)
                    buyPressure += volume * Math.Abs(priceChange);
                else
                    sellPressure += volume * Math.Abs(priceChange);
            }

            var totalPressure = buyPressure + sellPressure;
            if (totalPressure < 1e-10) return 0;

            return (buyPressure - sellPressure) / totalPressure;
        }

        private double CalculateVolumeMomentumCorrelation(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var priceChanges = new List<double>();
            var volumes = new List<double>();

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                priceChanges.Add((double)(bars[i].Close - bars[i - 1].Close));
                volumes.Add(bars[i].TickVolume);
            }

            // Calculate correlation
            var avgPrice = priceChanges.Average();
            var avgVolume = volumes.Average();

            double covariance = 0;
            double priceVar = 0;
            double volumeVar = 0;

            for (int i = 0; i < priceChanges.Count; i++)
            {
                var pDiff = priceChanges[i] - avgPrice;
                var vDiff = volumes[i] - avgVolume;

                covariance += pDiff * vDiff;
                priceVar += pDiff * pDiff;
                volumeVar += vDiff * vDiff;
            }

            return SafeDiv(covariance, Math.Sqrt(priceVar * volumeVar));
        }

        private double CalculateTrendStrength(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified ADX calculation
            double sumDM = 0;
            double sumTR = 0;

            for (int i = currentIndex - 13; i <= currentIndex; i++)
            {
                if (i > 0)
                {
                    var highDiff = (double)(bars[i].High - bars[i - 1].High);
                    var lowDiff = (double)(bars[i - 1].Low - bars[i].Low);

                    var dm = Math.Max(0, Math.Max(highDiff, lowDiff));
                    var tr = Math.Max((double)(bars[i].High - bars[i].Low),
                            Math.Max(Math.Abs((double)(bars[i].High - bars[i - 1].Close)),
                                    Math.Abs((double)(bars[i].Low - bars[i - 1].Close))));

                    sumDM += dm;
                    sumTR += tr;
                }
            }

            return Math.Min(1.0, SafeDiv(sumDM, sumTR));
        }

        private (double dmPlus, double dmMinus) CalculateDirectionalMovement(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            double sumDMPlus = 0;
            double sumDMMinus = 0;
            double sumTR = 0;

            for (int i = currentIndex - 13; i <= currentIndex; i++)
            {
                if (i > 0)
                {
                    var highDiff = (double)(bars[i].High - bars[i - 1].High);
                    var lowDiff = (double)(bars[i - 1].Low - bars[i].Low);

                    if (highDiff > lowDiff && highDiff > 0)
                        sumDMPlus += highDiff;
                    else if (lowDiff > highDiff && lowDiff > 0)
                        sumDMMinus += lowDiff;

                    var tr = Math.Max((double)(bars[i].High - bars[i].Low),
                            Math.Max(Math.Abs((double)(bars[i].High - bars[i - 1].Close)),
                                    Math.Abs((double)(bars[i].Low - bars[i - 1].Close))));

                    sumTR += tr;
                }
            }

            var dmPlus = SafeDiv(sumDMPlus, sumTR);
            var dmMinus = SafeDiv(sumDMMinus, sumTR);

            return (dmPlus, dmMinus);
        }

        private double CalculateTrendEfficiency(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            var direction = Math.Abs((double)(bars[currentIndex].Close - bars[currentIndex - period].Close));
            double volatility = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                volatility += Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
            }

            return SafeDiv(direction, volatility);
        }

        private (double supportDist, double resistanceDist) CalculateSRDistance(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var close = (double)bars[currentIndex].Close;
            var atr = CalculateATR(bars, currentIndex, 14);

            // Find recent support (lowest low in last 20 bars)
            double support = double.MaxValue;
            double resistance = double.MinValue;

            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                support = Math.Min(support, (double)bars[i].Low);
                resistance = Math.Max(resistance, (double)bars[i].High);
            }

            var supportDist = CalculateNormalizedDistance(close, support, atr);
            var resistanceDist = CalculateNormalizedDistance(close, resistance, atr);

            return (supportDist, resistanceDist);
        }

        private double CalculateBounceProbability(double supportDist, double resistanceDist)
        {
            // Near support = positive (bullish bounce)
            if (Math.Abs(supportDist) < 0.5) return 0.5 + supportDist;

            // Near resistance = negative (bearish bounce)
            if (Math.Abs(resistanceDist) < 0.5) return -0.5 + resistanceDist;

            return 0;
        }

        private double CalculateDirectionalProbability(double primary, double confirmation, double trendStrength)
        {
            // Weight factors based on market conditions
            var primaryWeight = 0.5 + trendStrength * 0.2;
            var confirmWeight = 0.3;

            var prob = primary * primaryWeight + confirmation * confirmWeight;

            // Apply sigmoid for smooth probability
            return Sigmoid(prob, 2.0);
        }

        private double CalculateMomentumDivergence(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var prices = new double[10];
            var momentum = new double[10];

            for (int i = 0; i < 10; i++)
            {
                var idx = currentIndex - 9 + i;
                prices[i] = (double)bars[idx].Close;
                momentum[i] = idx >= 5 ? (double)(bars[idx].Close - bars[idx - 5].Close) : 0;
            }

            return CalculateDivergence(prices, momentum, 10);
        }

        private double CalculateExhaustionSignal(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Check for overextended moves
            var momentum = (double)(bars[currentIndex].Close - bars[currentIndex - 10].Close);
            var atr = CalculateATR(bars, currentIndex, 14);

            var extension = SafeDiv(Math.Abs(momentum), atr);

            // Exhaustion if move > 3 ATR
            if (extension > 3)
                return -Math.Sign(momentum);  // Expect reversal

            return 0;
        }

        private double CalculateMeanReversionProbability(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var sma20 = CalculateSMA(bars, currentIndex, 20);
            var close = (double)bars[currentIndex].Close;
            var deviation = close - sma20;
            var atr = CalculateATR(bars, currentIndex, 14);

            var normalizedDev = SafeDiv(deviation, atr);

            // High probability of mean reversion if > 2 ATR from mean
            if (Math.Abs(normalizedDev) > 2)
                return -Math.Sign(normalizedDev) * Math.Min(1.0, Math.Abs(normalizedDev) / 3);

            return 0;
        }

        public override void Reset()
        {
            _priceHistory.Clear();
            _momentumHistory.Clear();
            _volumeHistory.Clear();
        }
    }
}