using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Technical indicators transformed and optimized for 3-class prediction
    /// All indicators are normalized and converted to directional signals
    /// </summary>
    public class TechnicalIndicatorFeatures : BaseFeatureCalculator
    {
        public override string Name => "TechnicalIndicators";
        public override string Category => "Technical";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 5;

        private readonly RollingWindow<double> _rsiHistory = new(50);
        private readonly RollingWindow<double> _macdHistory = new(50);
        private readonly RollingWindow<double> _stochHistory = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return;

            var close = (double)bars[currentIndex].Close;

            // ===== 1. RSI-BASED FEATURES (TRANSFORMED) =====

            // Standard RSI(14)
            var rsi14 = CalculateRSI(bars, currentIndex, 14);
            _rsiHistory.Add(rsi14);

            // RSI normalized to [-1, 1] for directionality
            var rsiNormalized = (rsi14 - 50) / 50;  // -1 = oversold, 1 = overbought
            output.AddFeature("tech_rsi_normalized", rsiNormalized);

            // RSI directional signal with dynamic thresholds
            var rsiSignal = CreateRSIDirectionalSignal(rsi14, _rsiHistory);
            output.AddFeature("tech_rsi_signal", rsiSignal);

            // RSI divergence (powerful reversal signal)
            var rsiDivergence = CalculateRSIDivergence(bars, currentIndex, rsi14);
            output.AddFeature("tech_rsi_divergence", rsiDivergence);

            // RSI momentum (rate of change)
            if (_rsiHistory.Count >= 5)
            {
                var rsiMomentum = (rsi14 - _rsiHistory[4]) / 5;
                output.AddFeature("tech_rsi_momentum", Sigmoid(rsiMomentum / 10));
            }

            // Multi-timeframe RSI composite
            var rsi9 = CalculateRSI(bars, currentIndex, 9);
            var rsi21 = CalculateRSI(bars, currentIndex, 21);
            var rsiComposite = (rsi9 * 0.3 + rsi14 * 0.4 + rsi21 * 0.3 - 50) / 50;
            output.AddFeature("tech_rsi_composite", rsiComposite);

            // ===== 2. MACD-BASED FEATURES (TRANSFORMED) =====

            var (macdLine, macdSignal, macdHist) = CalculateMACD(bars, currentIndex);
            _macdHistory.Add(macdHist);

            // MACD histogram normalized by ATR for comparability
            var atr = CalculateATR(bars, currentIndex, 14);
            var macdNormalized = SafeDiv(macdHist, atr);
            output.AddFeature("tech_macd_normalized", Sigmoid(macdNormalized));

            // MACD cross signal (strong directional indicator)
            var macdCross = DetectMACDCross(macdLine, macdSignal, _macdHistory);
            output.AddFeature("tech_macd_cross", macdCross);

            // MACD divergence
            var macdDivergence = CalculateMACDDivergence(bars, currentIndex, macdHist);
            output.AddFeature("tech_macd_divergence", macdDivergence);

            // MACD momentum quality
            var macdQuality = CalculateMACDQuality(_macdHistory);
            output.AddFeature("tech_macd_quality", macdQuality);

            // ===== 3. STOCHASTIC FEATURES (TRANSFORMED) =====

            var (stochK, stochD) = CalculateStochastic(bars, currentIndex, 14, 3);
            _stochHistory.Add(stochK);

            // Stochastic normalized to directional signal
            var stochNormalized = (stochK - 50) / 50;
            output.AddFeature("tech_stoch_normalized", stochNormalized);

            // Stochastic cross signal
            var stochCross = stochK > stochD && stochK > 20 && stochK < 80 ?
                            Math.Sign(stochK - 50) : 0;
            output.AddFeature("tech_stoch_cross", stochCross);

            // Stochastic divergence
            var stochDivergence = CalculateStochasticDivergence(bars, currentIndex, stochK);
            output.AddFeature("tech_stoch_divergence", stochDivergence);

            // ===== 4. BOLLINGER BANDS FEATURES (TRANSFORMED) =====

            var (bbUpper, bbMiddle, bbLower, bbWidth) = CalculateBollingerBands(bars, currentIndex, 20, 2);

            // BB position as directional signal
            var bbPosition = SafeDiv(close - bbLower, bbUpper - bbLower) * 2 - 1;  // [-1, 1]
            output.AddFeature("tech_bb_position", bbPosition);

            // BB squeeze detection (volatility contraction)
            var bbSqueeze = DetectBBSqueeze(bars, currentIndex, bbWidth);
            output.AddFeature("tech_bb_squeeze", bbSqueeze);

            // BB band touch signals
            var bbTouch = close > bbUpper ? 1.0 : close < bbLower ? -1.0 : 0.0;
            output.AddFeature("tech_bb_touch", bbTouch);

            // BB expansion signal (breakout potential)
            var bbExpansion = CalculateBBExpansion(bars, currentIndex, bbWidth);
            output.AddFeature("tech_bb_expansion", bbExpansion);

            // ===== 5. MOVING AVERAGE FEATURES (TRANSFORMED) =====

            var ema9 = CalculateEMA(bars, currentIndex, 9);
            var ema21 = CalculateEMA(bars, currentIndex, 21);
            var ema50 = CalculateEMA(bars, currentIndex, 50);

            // MA alignment signal (trend confirmation)
            var maAlignment = CalculateMAAlignment(close, ema9, ema21, ema50);
            output.AddFeature("tech_ma_alignment", maAlignment);

            // MA cross signals
            var maCross921 = DetectMACross(ema9, ema21, bars, currentIndex);
            output.AddFeature("tech_ma_cross_9_21", maCross921);

            var maCross2150 = DetectMACross(ema21, ema50, bars, currentIndex);
            output.AddFeature("tech_ma_cross_21_50", maCross2150);

            // Price-MA deviation (mean reversion)
            var maDev9 = SafeDiv(close - ema9, atr);
            output.AddFeature("tech_ma_dev_9", Sigmoid(maDev9));

            var maDev21 = SafeDiv(close - ema21, atr);
            output.AddFeature("tech_ma_dev_21", Sigmoid(maDev21));

            // MA slope convergence/divergence
            var maConvergence = CalculateMAConvergence(bars, currentIndex);
            output.AddFeature("tech_ma_convergence", maConvergence);

            // ===== 6. ATR/VOLATILITY FEATURES (TRANSFORMED) =====

            var atr14 = CalculateATR(bars, currentIndex, 14);
            var atr7 = CalculateATR(bars, currentIndex, 7);

            // ATR expansion ratio (volatility regime)
            var atrRatio = SafeDiv(atr7, atr14);
            output.AddFeature("tech_atr_ratio", atrRatio);

            // Volatility percentile
            var volPercentile = CalculateVolatilityPercentile(bars, currentIndex, atr14);
            output.AddFeature("tech_vol_percentile", volPercentile);

            // Volatility regime signal
            var volRegime = volPercentile > 0.7 ? 1.0 :   // High volatility
                           volPercentile < 0.3 ? -1.0 :   // Low volatility
                           0.0;                            // Normal
            output.AddFeature("tech_vol_regime", volRegime);

            // ===== 7. COMPOSITE TECHNICAL SIGNALS =====

            // Oscillator composite (RSI + Stochastic + MACD)
            var oscillatorComposite = CreateCompositeSignal(
                (rsiNormalized, 0.35),
                (stochNormalized, 0.35),
                (Sigmoid(macdNormalized), 0.30)
            );
            output.AddFeature("tech_oscillator_composite", oscillatorComposite);

            // Trend composite (MA alignment + MACD)
            var trendComposite = CreateCompositeSignal(
                (maAlignment, 0.4),
                (maCross921, 0.3),
                (Math.Sign(macdLine), 0.3)
            );
            output.AddFeature("tech_trend_composite", trendComposite);

            // Reversal composite (divergences + extremes)
            var reversalComposite = CreateCompositeSignal(
                (rsiDivergence, 0.3),
                (macdDivergence, 0.3),
                (stochDivergence, 0.2),
                (bbTouch * -1, 0.2)  // Band touch suggests reversal
            );
            output.AddFeature("tech_reversal_composite", reversalComposite);

            // Master technical signal
            var masterSignal = CreateCompositeSignal(
                (oscillatorComposite, 0.35),
                (trendComposite, 0.35),
                (reversalComposite * -0.3, 0.3)  // Reversal opposes trend
            );
            output.AddFeature("tech_master_signal", masterSignal);

            // Signal confidence based on agreement
            var signalAgreement = CalculateSignalAgreement(
                rsiNormalized, stochNormalized, Sigmoid(macdNormalized),
                maAlignment, bbPosition);
            output.AddFeature("tech_signal_confidence", signalAgreement);
        }

        // ===== CALCULATION METHODS =====

        private double CalculateRSI(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            if (currentIndex < period) return 50;

            double gains = 0, losses = 0;
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

        private double CreateRSIDirectionalSignal(double rsi, RollingWindow<double> history)
        {
            // Dynamic thresholds based on recent RSI range
            if (history.Count < 20)
                return CreateDirectionalSignal(rsi, 70, 30);

            var recentValues = history.GetValues().Take(20).ToList();
            var percentile80 = recentValues.OrderBy(x => x).Skip(16).First();
            var percentile20 = recentValues.OrderBy(x => x).Skip(4).First();

            return CreateDirectionalSignal(rsi, percentile80, percentile20);
        }

        private double CalculateRSIDivergence(IReadOnlyList<OhlcBar> bars, int currentIndex, double currentRSI)
        {
            if (_rsiHistory.Count < 10) return 0;

            var prices = new double[10];
            var rsiValues = new double[10];

            for (int i = 0; i < 10; i++)
            {
                prices[9 - i] = (double)bars[currentIndex - i].Close;
                rsiValues[9 - i] = i < _rsiHistory.Count ? _rsiHistory[i] : currentRSI;
            }

            return CalculateDivergence(prices, rsiValues, 10);
        }

        private (double line, double signal, double histogram) CalculateMACD(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var ema12 = CalculateEMA(bars, currentIndex, 12);
            var ema26 = CalculateEMA(bars, currentIndex, 26);
            var macdLine = ema12 - ema26;

            // Signal line (9-period EMA of MACD)
            var macdSignal = CalculateMACDSignal(bars, currentIndex);
            var histogram = macdLine - macdSignal;

            return (macdLine, macdSignal, histogram);
        }

        private double CalculateMACDSignal(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified: calculate MACD values for last 9 bars and average
            var macdValues = new List<double>();
            for (int i = currentIndex - 8; i <= currentIndex; i++)
            {
                if (i >= 26)
                {
                    var ema12 = CalculateEMA(bars, i, 12);
                    var ema26 = CalculateEMA(bars, i, 26);
                    macdValues.Add(ema12 - ema26);
                }
            }
            return macdValues.Count > 0 ? macdValues.Average() : 0;
        }

        private double DetectMACDCross(double line, double signal, RollingWindow<double> history)
        {
            if (history.Count < 2) return 0;

            var prevHist = history[1];
            var currentHist = line - signal;

            // Bullish cross
            if (prevHist <= 0 && currentHist > 0) return 1.0;

            // Bearish cross
            if (prevHist >= 0 && currentHist < 0) return -1.0;

            return 0;
        }

        private double CalculateMACDDivergence(IReadOnlyList<OhlcBar> bars, int currentIndex, double currentMACD)
        {
            if (_macdHistory.Count < 10) return 0;

            var prices = new double[10];
            var macdValues = new double[10];

            for (int i = 0; i < 10; i++)
            {
                prices[9 - i] = (double)bars[currentIndex - i].Close;
                macdValues[9 - i] = i < _macdHistory.Count ? _macdHistory[i] : currentMACD;
            }

            return CalculateDivergence(prices, macdValues, 10);
        }

        private double CalculateMACDQuality(RollingWindow<double> history)
        {
            if (history.Count < 5) return 0;

            var recent = history.GetValues().Take(5).ToList();
            return CalculateMomentumQuality(recent);
        }

        private (double k, double d) CalculateStochastic(IReadOnlyList<OhlcBar> bars, int currentIndex, int period, int smoothK)
        {
            if (currentIndex < period) return (50, 50);

            double highest = double.MinValue;
            double lowest = double.MaxValue;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                highest = Math.Max(highest, (double)bars[i].High);
                lowest = Math.Min(lowest, (double)bars[i].Low);
            }

            var close = (double)bars[currentIndex].Close;
            var k = SafeDiv(close - lowest, highest - lowest) * 100;

            // %D is SMA of %K
            var d = k;  // Simplified
            if (currentIndex >= period + smoothK)
            {
                var kValues = new List<double>();
                for (int i = 0; i < smoothK; i++)
                {
                    // Recalculate K for previous bars
                    kValues.Add(k);  // Simplified - should calculate actual K values
                }
                d = kValues.Average();
            }

            return (k, d);
        }

        private double CalculateStochasticDivergence(IReadOnlyList<OhlcBar> bars, int currentIndex, double currentStoch)
        {
            if (_stochHistory.Count < 10) return 0;

            var prices = new double[10];
            var stochValues = new double[10];

            for (int i = 0; i < 10; i++)
            {
                prices[9 - i] = (double)bars[currentIndex - i].Close;
                stochValues[9 - i] = i < _stochHistory.Count ? _stochHistory[i] : currentStoch;
            }

            return CalculateDivergence(prices, stochValues, 10);
        }

        private (double upper, double middle, double lower, double width) CalculateBollingerBands(
            IReadOnlyList<OhlcBar> bars, int currentIndex, int period, double stdMult)
        {
            var sma = CalculateSMA(bars, currentIndex, period);
            var stdDev = CalculateStdDev(bars, currentIndex, period);

            var upper = sma + stdMult * stdDev;
            var lower = sma - stdMult * stdDev;
            var width = upper - lower;

            return (upper, sma, lower, width);
        }

        private double DetectBBSqueeze(IReadOnlyList<OhlcBar> bars, int currentIndex, double currentWidth)
        {
            // Calculate historical BB width
            var historicalWidths = new List<double>();
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                if (i >= 20)
                {
                    var (_, _, _, width) = CalculateBollingerBands(bars, i, 20, 2);
                    historicalWidths.Add(width);
                }
            }

            if (historicalWidths.Count < 10) return 0;

            var percentile = CalculatePercentileRank(currentWidth, historicalWidths);
            return percentile < 20 ? 1.0 : 0.0;  // Squeeze if width in bottom 20%
        }

        private double CalculateBBExpansion(IReadOnlyList<OhlcBar> bars, int currentIndex, double currentWidth)
        {
            if (currentIndex < 25) return 0;

            var (_, _, _, prevWidth) = CalculateBollingerBands(bars, currentIndex - 5, 20, 2);
            var expansion = SafeDiv(currentWidth - prevWidth, prevWidth);

            return Sigmoid(expansion * 100);  // Normalized expansion signal
        }

        private double CalculateEMA(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            if (currentIndex < period - 1) return (double)bars[currentIndex].Close;

            var multiplier = 2.0 / (period + 1);

            // Initialize with SMA
            double ema = 0;
            for (int i = currentIndex - period + 1; i <= currentIndex - period + 1 + period - 1; i++)
            {
                ema += (double)bars[i].Close;
            }
            ema /= period;

            // Calculate EMA
            for (int i = currentIndex - period + 1 + period; i <= currentIndex; i++)
            {
                ema = ((double)bars[i].Close - ema) * multiplier + ema;
            }

            return ema;
        }

        private double CalculateSMA(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            double sum = 0;
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                sum += (double)bars[i].Close;
            }
            return sum / period;
        }

        private double CalculateStdDev(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            var mean = CalculateSMA(bars, currentIndex, period);
            double sumSquares = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var diff = (double)bars[i].Close - mean;
                sumSquares += diff * diff;
            }

            return Math.Sqrt(sumSquares / period);
        }

        private double CalculateMAAlignment(double price, double ema9, double ema21, double ema50)
        {
            // Perfect bullish alignment: price > EMA9 > EMA21 > EMA50
            if (price > ema9 && ema9 > ema21 && ema21 > ema50)
                return 1.0;

            // Perfect bearish alignment: price < EMA9 < EMA21 < EMA50
            if (price < ema9 && ema9 < ema21 && ema21 < ema50)
                return -1.0;

            // Partial alignment
            int bullishCount = 0;
            if (price > ema9) bullishCount++;
            if (ema9 > ema21) bullishCount++;
            if (ema21 > ema50) bullishCount++;

            return (bullishCount - 1.5) / 1.5;  // Normalize to [-1, 1]
        }

        private double DetectMACross(double fastMA, double slowMA, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return 0;

            var prevFast = CalculateEMA(bars, currentIndex - 1,
                fastMA == CalculateEMA(bars, currentIndex, 9) ? 9 : 21);
            var prevSlow = CalculateEMA(bars, currentIndex - 1,
                slowMA == CalculateEMA(bars, currentIndex, 21) ? 21 : 50);

            // Golden cross
            if (prevFast <= prevSlow && fastMA > slowMA) return 1.0;

            // Death cross
            if (prevFast >= prevSlow && fastMA < slowMA) return -1.0;

            return 0;
        }

        private double CalculateMAConvergence(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var ema9 = CalculateEMA(bars, currentIndex, 9);
            var ema21 = CalculateEMA(bars, currentIndex, 21);
            var ema50 = CalculateEMA(bars, currentIndex, 50);

            var spread1 = Math.Abs(ema9 - ema21);
            var spread2 = Math.Abs(ema21 - ema50);

            // Previous spreads
            if (currentIndex >= 5)
            {
                var prevEma9 = CalculateEMA(bars, currentIndex - 5, 9);
                var prevEma21 = CalculateEMA(bars, currentIndex - 5, 21);
                var prevEma50 = CalculateEMA(bars, currentIndex - 5, 50);

                var prevSpread1 = Math.Abs(prevEma9 - prevEma21);
                var prevSpread2 = Math.Abs(prevEma21 - prevEma50);

                // Convergence if spreads decreasing
                var convergence1 = SafeDiv(prevSpread1 - spread1, prevSpread1);
                var convergence2 = SafeDiv(prevSpread2 - spread2, prevSpread2);

                return Sigmoid((convergence1 + convergence2) * 50);
            }

            return 0;
        }

        private double CalculateATR(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            double sum = 0;
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var tr = Math.Max((double)(bars[i].High - bars[i].Low),
                        Math.Max(Math.Abs((double)(bars[i].High - bars[i - 1].Close)),
                                Math.Abs((double)(bars[i].Low - bars[i - 1].Close))));
                sum += tr;
            }
            return sum / period;
        }

        private double CalculateVolatilityPercentile(IReadOnlyList<OhlcBar> bars, int currentIndex, double currentATR)
        {
            var historicalATRs = new List<double>();
            for (int i = currentIndex - 49; i <= currentIndex; i++)
            {
                if (i >= 14)
                {
                    historicalATRs.Add(CalculateATR(bars, i, 14));
                }
            }

            return CalculatePercentileRank(currentATR, historicalATRs) / 100;
        }

        private double CalculateSignalAgreement(params double[] signals)
        {
            if (signals.Length == 0) return 0;

            var positiveCount = signals.Count(s => s > 0.2);
            var negativeCount = signals.Count(s => s < -0.2);

            // High agreement if most signals point same direction
            if (positiveCount > signals.Length * 0.7)
                return (double)positiveCount / signals.Length;

            if (negativeCount > signals.Length * 0.7)
                return -(double)negativeCount / signals.Length;

            return 0;  // No clear agreement
        }

        public override void Reset()
        {
            _rsiHistory.Clear();
            _macdHistory.Clear();
            _stochHistory.Clear();
        }
    }
}