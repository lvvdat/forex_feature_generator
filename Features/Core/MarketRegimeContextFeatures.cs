using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Market regime and context features for adaptive prediction
    /// Different market conditions require different feature interpretations
    /// </summary>
    public class MarketRegimeContextFeatures : BaseFeatureCalculator
    {
        public override string Name => "MarketRegimeContext";
        public override string Category => "Context";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 2;

        private readonly RollingWindow<RegimeSnapshot> _regimeHistory = new(100);
        private readonly RollingWindow<double> _volatilityHistory = new(50);

        private class RegimeSnapshot
        {
            public DateTime Timestamp { get; set; }
            public int RegimeType { get; set; } // 0=Range, 1=Trend, 2=Volatile
            public double Volatility { get; set; }
            public double TrendStrength { get; set; }
            public double Efficiency { get; set; }
        }

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return;

            var close = (double)bars[currentIndex].Close;

            // ===== 1. MARKET REGIME DETECTION =====
            // Critical for adapting predictions to market conditions

            var (regimeType, regimeConfidence) = DetectMarketRegime(bars, currentIndex);
            output.AddFeature("regime_type", regimeType);
            output.AddFeature("regime_confidence", regimeConfidence);

            // Regime-specific directional bias
            var regimeBias = CalculateRegimeBias(regimeType, bars, currentIndex);
            output.AddFeature("regime_directional_bias", regimeBias);

            // Regime transition probability
            var transitionProb = CalculateRegimeTransition(_regimeHistory);
            output.AddFeature("regime_transition_prob", transitionProb);

            // Regime duration and stability
            var (duration, stability) = CalculateRegimeStability(_regimeHistory, regimeType);
            output.AddFeature("regime_duration_norm", Sigmoid(duration / 20.0));
            output.AddFeature("regime_stability", stability);

            // ===== 2. VOLATILITY REGIME FEATURES =====
            // Volatility affects label probability distributions

            var currentVol = CalculateRealizedVolatility(bars, currentIndex, 20);
            _volatilityHistory.Add(currentVol);

            // Volatility regime classification
            var volRegime = ClassifyVolatilityRegime(currentVol, _volatilityHistory);
            output.AddFeature("vol_regime_type", volRegime);

            // Volatility trend (expanding/contracting)
            var volTrend = CalculateVolatilityTrend(_volatilityHistory);
            output.AddFeature("vol_trend", volTrend);
            output.AddFeature("vol_expansion_signal", volTrend > 0.3 ? 1.0 : volTrend < -0.3 ? -1.0 : 0.0);

            // GARCH volatility forecast
            var garchVol = CalculateGARCHVolatility(bars, currentIndex);
            output.AddFeature("vol_garch_forecast", garchVol);

            // Volatility surprise (actual vs expected)
            var volSurprise = currentVol - garchVol;
            output.AddFeature("vol_surprise", Sigmoid(volSurprise * 100));

            // ===== 3. TREND REGIME FEATURES =====
            // Trend characteristics affect directional probabilities

            // Multi-timeframe trend alignment
            var (trendAlignment, trendStrength) = CalculateMultiTimeframeTrend(bars, currentIndex);
            output.AddFeature("trend_mtf_alignment", trendAlignment);
            output.AddFeature("trend_mtf_strength", trendStrength);

            // Trend efficiency (Kaufman)
            var efficiency = CalculateKaufmanEfficiency(bars, currentIndex, 10);
            output.AddFeature("trend_efficiency", efficiency);

            // Trend quality metrics
            var trendQuality = CalculateTrendQuality(bars, currentIndex);
            output.AddFeature("trend_quality", trendQuality);

            // Trend exhaustion signals
            var exhaustion = DetectTrendExhaustion(bars, currentIndex);
            output.AddFeature("trend_exhaustion", exhaustion);

            // ===== 4. CYCLICAL & SEASONAL PATTERNS =====
            // Time-based patterns affect direction

            var bar = bars[currentIndex];

            // Intraday patterns (hour of day effects)
            var hourOfDay = bar.Timestamp.Hour;
            var sessionType = GetTradingSession(hourOfDay);
            output.AddFeature("session_type", sessionType);

            // Session-specific volatility
            var sessionVol = CalculateSessionVolatility(hourOfDay, bars, currentIndex);
            output.AddFeature("session_volatility_ratio", sessionVol);

            // Day of week effects
            var dayOfWeek = (int)bar.Timestamp.DayOfWeek;
            var dayEffect = CalculateDayOfWeekEffect(dayOfWeek, bars, currentIndex);
            output.AddFeature("day_effect", dayEffect);

            // Cyclical components (simplified)
            var cyclicalPhase = CalculateCyclicalPhase(bars, currentIndex);
            output.AddFeature("cyclical_phase", cyclicalPhase);

            // ===== 5. MARKET STRESS INDICATORS =====
            // Stress conditions affect prediction reliability

            // Stress index composite
            var stressIndex = CalculateMarketStress(bars, currentIndex);
            output.AddFeature("market_stress", stressIndex);

            // Risk on/off sentiment
            var riskSentiment = CalculateRiskSentiment(bars, currentIndex);
            output.AddFeature("risk_sentiment", riskSentiment);

            // Correlation breakdown detection
            var corrBreakdown = DetectCorrelationBreakdown(bars, currentIndex);
            output.AddFeature("correlation_breakdown", corrBreakdown);

            // ===== 6. FRACTAL & CHAOS FEATURES =====
            // Non-linear dynamics for complex markets

            // Fractal dimension (market complexity)
            var fractalDim = CalculateFractalDimension(bars, currentIndex);
            output.AddFeature("fractal_dimension", (fractalDim - 1.5) / 0.5); // Normalize around 1.5

            // Hurst exponent (persistence/mean-reversion)
            var hurst = CalculateHurstExponent(bars, currentIndex);
            output.AddFeature("hurst_exponent", (hurst - 0.5) * 2); // Normalize: <0 mean-reverting, >0 trending

            // Lyapunov exponent proxy (chaos indicator)
            var lyapunov = CalculateLyapunovProxy(bars, currentIndex);
            output.AddFeature("chaos_indicator", lyapunov);

            // ===== 7. REGIME-ADAPTIVE SIGNALS =====
            // Combine regime information with directional indicators

            // Regime-weighted momentum
            var adaptiveMomentum = CalculateAdaptiveMomentum(bars, currentIndex, regimeType);
            output.AddFeature("regime_momentum", adaptiveMomentum);

            // Regime-specific reversal probability
            var reversalProb = CalculateRegimeReversalProbability(
                regimeType, efficiency, trendStrength, exhaustion);
            output.AddFeature("regime_reversal_prob", reversalProb);

            // Regime-adjusted directional signal
            var regimeSignal = CalculateRegimeAdjustedSignal(
                bars, currentIndex, regimeType, volRegime);
            output.AddFeature("regime_directional_signal", regimeSignal);

            // ===== 8. COMPOSITE REGIME INDICATORS =====

            // Market condition score (favorable for trading)
            var marketCondition = CalculateMarketConditionScore(
                regimeConfidence, trendQuality, efficiency, stressIndex);
            output.AddFeature("market_condition_score", marketCondition);

            // Predictability index
            var predictability = CalculatePredictabilityIndex(
                hurst, efficiency, regimeConfidence, corrBreakdown);
            output.AddFeature("predictability_index", predictability);

            // Master regime signal
            var masterRegime = CreateCompositeSignal(
                (regimeBias, 0.25),
                (adaptiveMomentum, 0.25),
                (riskSentiment, 0.2),
                (regimeSignal, 0.3)
            );
            output.AddFeature("regime_master_signal", masterRegime);

            // Update history
            _regimeHistory.Add(new RegimeSnapshot
            {
                Timestamp = bar.Timestamp,
                RegimeType = (int)regimeType,
                Volatility = currentVol,
                TrendStrength = trendStrength,
                Efficiency = efficiency
            });
        }

        // ===== REGIME DETECTION METHODS =====

        private (double type, double confidence) DetectMarketRegime(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var volatility = CalculateRealizedVolatility(bars, currentIndex, 14);
            var avgVolatility = CalculateRealizedVolatility(bars, currentIndex, 50);

            var efficiency = CalculateKaufmanEfficiency(bars, currentIndex, 20);
            var adx = CalculateADX(bars, currentIndex, 14);

            // Classify regime
            double regimeType;
            double confidence;

            if (volatility > avgVolatility * 1.5)
            {
                regimeType = 2; // Volatile regime
                confidence = Math.Min(1.0, volatility / (avgVolatility * 2));
            }
            else if (adx > 25 && efficiency > 0.3)
            {
                regimeType = 1; // Trending regime
                confidence = Math.Min(1.0, (adx - 20) / 30.0) * efficiency;
            }
            else
            {
                regimeType = 0; // Range-bound regime
                confidence = 1.0 - efficiency;
            }

            return (regimeType, confidence);
        }

        private double CalculateRegimeBias(double regimeType, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (regimeType == 1) // Trending
            {
                // Follow trend direction
                var trend = CalculateTrendDirection(bars, currentIndex, 20);
                return trend;
            }
            else if (regimeType == 0) // Range-bound
            {
                // Mean reversion bias
                var meanReversion = CalculateMeanReversionSignal(bars, currentIndex);
                return meanReversion;
            }

            return 0; // No bias in volatile regime
        }

        private double CalculateRegimeTransition(RollingWindow<RegimeSnapshot> history)
        {
            if (history.Count < 20) return 0;

            var regimes = history.GetValues().Take(20).Select(r => r.RegimeType).ToList();
            int transitions = 0;

            for (int i = 1; i < regimes.Count; i++)
            {
                if (regimes[i] != regimes[i - 1])
                    transitions++;
            }

            return (double)transitions / regimes.Count;
        }

        private (double duration, double stability) CalculateRegimeStability(
            RollingWindow<RegimeSnapshot> history, double currentRegime)
        {
            if (history.Count == 0) return (1, 0.5);

            int duration = 1;
            foreach (var snapshot in history.GetValues())
            {
                if (Math.Abs(snapshot.RegimeType - currentRegime) < 0.1)
                    duration++;
                else
                    break;
            }

            // Stability based on regime consistency
            var regimes = history.GetValues().Take(20).Select(r => r.RegimeType).ToList();
            var avgRegime = regimes.Average();
            var variance = regimes.Select(r => Math.Pow(r - avgRegime, 2)).Average();
            var stability = 1.0 / (1.0 + Math.Sqrt(variance));

            return (duration, stability);
        }

        // ===== VOLATILITY METHODS =====

        private double CalculateRealizedVolatility(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            if (currentIndex < period) return 0;

            var returns = new List<double>();
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var logReturn = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                returns.Add(logReturn);
            }

            var variance = returns.Select(r => r * r).Average();
            return Math.Sqrt(variance * 252 * 1440);  // Annualized for 1-min bars
        }

        private double ClassifyVolatilityRegime(double currentVol, RollingWindow<double> history)
        {
            if (history.Count < 30) return 0;

            var percentile = CalculatePercentileRank(currentVol, history.GetValues().Take(30).ToList());

            if (percentile > 80) return 1.0;   // High volatility
            if (percentile < 20) return -1.0;  // Low volatility
            return 0.0;                        // Normal volatility
        }

        private double CalculateVolatilityTrend(RollingWindow<double> history)
        {
            if (history.Count < 10) return 0;

            var values = history.GetValues().Take(10).Reverse().ToArray();
            var slope = CalculateSlope(values);

            return Sigmoid(slope * 1000);
        }

        private double CalculateGARCHVolatility(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified GARCH(1,1)
            const double omega = 0.000001;
            const double alpha = 0.05;
            const double beta = 0.94;

            var returns = new List<double>();
            for (int i = Math.Max(1, currentIndex - 29); i <= currentIndex; i++)
            {
                returns.Add(Math.Log((double)bars[i].Close / (double)bars[i - 1].Close));
            }

            var unconditionalVar = returns.Select(r => r * r).Average();
            double garchVar = unconditionalVar;

            foreach (var ret in returns)
            {
                garchVar = omega + alpha * ret * ret + beta * garchVar;
            }

            return Math.Sqrt(garchVar * 252 * 1440);
        }

        // ===== TREND METHODS =====

        private (double alignment, double strength) CalculateMultiTimeframeTrend(
            IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Short, medium, long term trends
            var trend5 = CalculateTrendDirection(bars, currentIndex, 5);
            var trend20 = CalculateTrendDirection(bars, currentIndex, 20);
            var trend50 = currentIndex >= 50 ? CalculateTrendDirection(bars, currentIndex, 50) : trend20;

            // Alignment: all trends same direction
            var alignment = 0.0;
            if (Math.Sign(trend5) == Math.Sign(trend20) && Math.Sign(trend20) == Math.Sign(trend50))
            {
                alignment = Math.Sign(trend5);
            }
            else if (Math.Sign(trend5) == Math.Sign(trend20))
            {
                alignment = Math.Sign(trend5) * 0.5;
            }

            // Strength: average of absolute trends
            var strength = (Math.Abs(trend5) + Math.Abs(trend20) + Math.Abs(trend50)) / 3;

            return (alignment, strength);
        }

        private double CalculateTrendDirection(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            if (currentIndex < period) return 0;

            var startPrice = (double)bars[currentIndex - period + 1].Close;
            var endPrice = (double)bars[currentIndex].Close;

            var direction = (endPrice - startPrice) / startPrice;
            return Sigmoid(direction * 10000);  // Normalize pips to [-1, 1]
        }

        private double CalculateKaufmanEfficiency(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            if (currentIndex < period) return 0;

            var direction = Math.Abs((double)(bars[currentIndex].Close -
                                             bars[currentIndex - period + 1].Close));
            double volatility = 0;

            for (int i = currentIndex - period + 2; i <= currentIndex; i++)
            {
                volatility += Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
            }

            return SafeDiv(direction, volatility);
        }

        private double CalculateTrendQuality(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return 0.5;

            // Count bars in same direction
            int upBars = 0, downBars = 0;
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                if (bars[i].Close > bars[i].Open) upBars++;
                else downBars++;
            }

            // Quality based on consistency
            var consistency = Math.Abs(upBars - downBars) / 20.0;

            // Smoothness (low noise)
            var smoothness = CalculateKaufmanEfficiency(bars, currentIndex, 10);

            return (consistency + smoothness) / 2;
        }

        private double DetectTrendExhaustion(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 30) return 0;

            // Check for overextension
            var sma20 = CalculateSMA(bars, currentIndex, 20);
            var close = (double)bars[currentIndex].Close;
            var deviation = (close - sma20) / sma20;

            // Exhaustion if > 2% from mean
            if (Math.Abs(deviation) > 0.02)
            {
                // Check momentum weakening
                var momentum5 = (double)(bars[currentIndex].Close - bars[currentIndex - 5].Close);
                var momentum10 = (double)(bars[currentIndex - 5].Close - bars[currentIndex - 10].Close);

                if (Math.Abs(momentum5) < Math.Abs(momentum10) * 0.5)
                    return -Math.Sign(deviation);  // Exhaustion signal
            }

            return 0;
        }

        // ===== TIME & SEASONAL METHODS =====

        private double GetTradingSession(int hour)
        {
            // Asian: 0-8, London: 8-16, NY: 13-22, Overlap: 13-16
            if (hour >= 13 && hour <= 16) return 2.0;  // London-NY overlap (high activity)
            if (hour >= 8 && hour <= 22) return 1.0;   // Major sessions
            return 0.0;  // Asian session
        }

        private double CalculateSessionVolatility(int hour, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Compare current volatility to typical for this session
            var currentVol = CalculateRealizedVolatility(bars, currentIndex, 10);

            // Typical session volatility (simplified)
            var typicalVol = hour >= 13 && hour <= 16 ? 0.001 :  // High during overlap
                            hour >= 8 && hour <= 22 ? 0.0007 :   // Normal during major sessions
                            0.0004;  // Low during Asian

            return SafeDiv(currentVol, typicalVol) - 1;  // Ratio to typical
        }

        private double CalculateDayOfWeekEffect(int dayOfWeek, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Monday/Friday effects
            if (dayOfWeek == 1) return -0.2;  // Monday: cautious
            if (dayOfWeek == 5) return 0.2;   // Friday: position closing
            return 0;
        }

        private double CalculateCyclicalPhase(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified cycle detection using sine wave fitting
            var period = 20;
            var phase = ((currentIndex % period) / (double)period) * 2 * Math.PI;
            return Math.Sin(phase);
        }

        // ===== MARKET STRESS METHODS =====

        private double CalculateMarketStress(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Composite stress indicator
            var volatility = CalculateRealizedVolatility(bars, currentIndex, 10);
            var avgVolatility = CalculateRealizedVolatility(bars, currentIndex, 50);
            var volStress = Math.Max(0, (volatility - avgVolatility) / avgVolatility);

            // Spread stress
            var currentSpread = (double)bars[currentIndex].AvgSpread;
            var avgSpread = 0.0;
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                avgSpread += (double)bars[i].AvgSpread;
            }
            avgSpread /= 20;
            var spreadStress = Math.Max(0, (currentSpread - avgSpread) / avgSpread);

            // Volume stress (abnormal volume)
            var volumeStress = 0.0;
            if (currentIndex >= 20)
            {
                var currentVolume = bars[currentIndex].TickVolume;
                var avgVolume = 0;
                for (int i = currentIndex - 19; i <= currentIndex - 1; i++)
                {
                    avgVolume += bars[i].TickVolume;
                }
                avgVolume /= 19;
                volumeStress = Math.Max(0, ((double)currentVolume - avgVolume) / avgVolume - 1);
            }

            return Sigmoid((volStress * 0.4 + spreadStress * 0.3 + volumeStress * 0.3) * 2);
        }

        private double CalculateRiskSentiment(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Risk-on: trending, normal volatility
            // Risk-off: ranging, high volatility

            var efficiency = CalculateKaufmanEfficiency(bars, currentIndex, 20);
            var volRegime = ClassifyVolatilityRegime(
                CalculateRealizedVolatility(bars, currentIndex, 14),
                _volatilityHistory);

            if (efficiency > 0.3 && volRegime <= 0)
                return 1.0;  // Risk-on
            if (efficiency < 0.2 && volRegime > 0)
                return -1.0;  // Risk-off

            return 0;
        }

        private double DetectCorrelationBreakdown(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Detect if normal price-volume correlations breaking down
            if (currentIndex < 20) return 0;

            var priceChanges = new List<double>();
            var volumes = new List<double>();

            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                priceChanges.Add(Math.Abs((double)(bars[i].Close - bars[i - 1].Close)));
                volumes.Add(bars[i].TickVolume);
            }

            // Calculate rolling correlation
            var correlation = CalculateCorrelation(priceChanges.ToArray(),
                                                  volumes.Select(v => (double)v).ToArray());

            // Breakdown if correlation becomes negative (unusual)
            return correlation < -0.2 ? 1.0 : 0.0;
        }

        // ===== FRACTAL & CHAOS METHODS =====

        private double CalculateFractalDimension(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified box-counting dimension
            if (currentIndex < 50) return 1.5;

            var prices = new double[50];
            for (int i = 0; i < 50; i++)
            {
                prices[i] = (double)bars[currentIndex - 49 + i].Close;
            }

            var maxPrice = prices.Max();
            var minPrice = prices.Min();
            var range = maxPrice - minPrice;

            if (range < 1e-10) return 1.5;

            // Count boxes at different scales
            int[] boxSizes = { 2, 5, 10, 25 };
            var boxCounts = new List<double>();

            foreach (var size in boxSizes)
            {
                var boxes = new HashSet<(int, int)>();
                var step = 50 / size;

                for (int i = 0; i < 49; i++)
                {
                    var x = i / step;
                    var y = (int)((prices[i] - minPrice) / range * size);
                    boxes.Add((x, y));
                }

                boxCounts.Add(boxes.Count);
            }

            // Log-log regression for fractal dimension
            var logSizes = boxSizes.Select(s => Math.Log(s)).ToArray();
            var logCounts = boxCounts.Select(c => Math.Log(c)).ToArray();

            var slope = -CalculateSlope(logCounts.Zip(logSizes, (c, s) => new { c, s })
                                              .OrderBy(p => p.s)
                                              .Select(p => p.c)
                                              .ToArray());

            return Math.Max(1.0, Math.Min(2.0, slope));
        }

        private double CalculateHurstExponent(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified R/S analysis
            if (currentIndex < 50) return 0.5;

            var returns = new double[50];
            for (int i = 1; i <= 50; i++)
            {
                returns[i - 1] = Math.Log((double)bars[currentIndex - 50 + i].Close /
                                         (double)bars[currentIndex - 50 + i - 1].Close);
            }

            var mean = returns.Average();
            var cumDev = new double[50];
            cumDev[0] = returns[0] - mean;

            for (int i = 1; i < 50; i++)
            {
                cumDev[i] = cumDev[i - 1] + returns[i] - mean;
            }

            var range = cumDev.Max() - cumDev.Min();
            var stdDev = Math.Sqrt(returns.Select(r => Math.Pow(r - mean, 2)).Average());

            if (stdDev < 1e-10) return 0.5;

            var rs = range / stdDev;
            var hurst = Math.Log(rs) / Math.Log(50);

            return Math.Max(0, Math.Min(1, hurst));
        }

        private double CalculateLyapunovProxy(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified Lyapunov exponent proxy
            if (currentIndex < 30) return 0;

            double sum = 0;
            for (int i = currentIndex - 29; i <= currentIndex - 1; i++)
            {
                var r1 = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                var r2 = Math.Log((double)bars[i + 1].Close / (double)bars[i].Close);

                if (Math.Abs(r1) > 1e-10)
                {
                    sum += Math.Log(Math.Abs(r2 / r1));
                }
            }

            return Sigmoid(sum / 29);
        }

        // ===== ADAPTIVE METHODS =====

        private double CalculateAdaptiveMomentum(IReadOnlyList<OhlcBar> bars, int currentIndex, double regimeType)
        {
            double momentum;

            if (regimeType == 1)  // Trending
            {
                // Use longer-term momentum
                momentum = (double)(bars[currentIndex].Close - bars[currentIndex - 20].Close);
            }
            else if (regimeType == 0)  // Range-bound
            {
                // Use short-term momentum with mean reversion
                momentum = -((double)bars[currentIndex].Close - CalculateSMA(bars, currentIndex, 10));
            }
            else  // Volatile
            {
                // Use very short-term momentum
                momentum = (double)(bars[currentIndex].Close - bars[currentIndex - 5].Close);
            }

            return Sigmoid(momentum * 10000);
        }

        private double CalculateMeanReversionSignal(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var sma = CalculateSMA(bars, currentIndex, 20);
            var close = (double)bars[currentIndex].Close;
            var deviation = (close - sma) / sma;

            // Mean reversion signal stronger at extremes
            if (Math.Abs(deviation) > 0.01)  // 1% from mean
            {
                return -Math.Sign(deviation) * Math.Min(1.0, Math.Abs(deviation) * 100);
            }

            return 0;
        }

        private double CalculateRegimeReversalProbability(double regime, double efficiency,
            double trendStrength, double exhaustion)
        {
            if (regime == 0)  // Range-bound
            {
                // High reversal probability at range extremes
                return 0.5 + exhaustion * 0.3;
            }
            else if (regime == 1)  // Trending
            {
                // Low reversal probability unless exhausted
                return Math.Max(0, exhaustion * 0.5 - trendStrength * 0.3);
            }
            else  // Volatile
            {
                // Moderate reversal probability
                return 0.3;
            }
        }

        private double CalculateRegimeAdjustedSignal(IReadOnlyList<OhlcBar> bars, int currentIndex,
            double regimeType, double volRegime)
        {
            var baseMomentum = (double)(bars[currentIndex].Close - bars[currentIndex - 10].Close);
            var signal = Sigmoid(baseMomentum * 10000);

            // Adjust based on regime
            if (regimeType == 0)  // Range-bound
            {
                signal *= -0.5;  // Fade moves
            }
            else if (regimeType == 1)  // Trending
            {
                signal *= 1.5;  // Follow trend
            }

            // Adjust for volatility
            if (volRegime > 0)  // High volatility
            {
                signal *= 0.7;  // Reduce confidence
            }

            return Math.Max(-1, Math.Min(1, signal));
        }

        private double CalculateMarketConditionScore(double regimeConf, double trendQuality,
            double efficiency, double stress)
        {
            // Good conditions: high confidence, quality, efficiency, low stress
            return (regimeConf * 0.2 + trendQuality * 0.3 + efficiency * 0.3 + (1 - stress) * 0.2);
        }

        private double CalculatePredictabilityIndex(double hurst, double efficiency,
            double regimeConf, double corrBreakdown)
        {
            // High predictability: trending (hurst > 0.5), efficient, stable regime, normal correlations
            var hurstScore = hurst > 0.5 ? hurst : 1 - hurst;  // Both trending and mean-reverting are predictable
            return (hurstScore * 0.3 + efficiency * 0.3 + regimeConf * 0.2 + (1 - corrBreakdown) * 0.2);
        }

        // ===== HELPER METHODS =====

        private double CalculateSMA(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            double sum = 0;
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                sum += (double)bars[i].Close;
            }
            return sum / period;
        }

        private double CalculateADX(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            // Simplified ADX
            if (currentIndex < period + 1) return 0;

            double sumDMPlus = 0, sumDMMinus = 0, sumTR = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
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

            var diPlus = SafeDiv(sumDMPlus, sumTR) * 100;
            var diMinus = SafeDiv(sumDMMinus, sumTR) * 100;
            var dx = SafeDiv(Math.Abs(diPlus - diMinus), diPlus + diMinus) * 100;

            return dx;
        }

        private double CalculateCorrelation(double[] x, double[] y)
        {
            if (x.Length != y.Length || x.Length < 2) return 0;

            var avgX = x.Average();
            var avgY = y.Average();

            double covariance = 0, varX = 0, varY = 0;

            for (int i = 0; i < x.Length; i++)
            {
                var diffX = x[i] - avgX;
                var diffY = y[i] - avgY;

                covariance += diffX * diffY;
                varX += diffX * diffX;
                varY += diffY * diffY;
            }

            return SafeDiv(covariance, Math.Sqrt(varX * varY));
        }

        public override void Reset()
        {
            _regimeHistory.Clear();
            _volatilityHistory.Clear();
        }
    }
}