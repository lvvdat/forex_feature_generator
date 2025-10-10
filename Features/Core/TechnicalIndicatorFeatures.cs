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
        public override int Priority => 4;

        private readonly RollingWindow<double> _rsiHistory = new(50);
        private readonly RollingWindow<double> _macdHistory = new(50);

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
            output.AddFeature("04_tech_rsi_normalized", rsiNormalized);

            // RSI momentum (rate of change)
            if (_rsiHistory.Count >= 5)
            {
                var rsiMomentum = (rsi14 - _rsiHistory[4]) / 5;
                output.AddFeature("04_tech_rsi_momentum", Sigmoid(rsiMomentum / 10));
            }
            else
            {
                output.AddFeature("04_tech_rsi_momentum", 0.0);
            }

            // Multi-timeframe RSI composite
            var rsi9 = CalculateRSI(bars, currentIndex, 9);
            var rsi21 = CalculateRSI(bars, currentIndex, 21);
            var rsiComposite = (rsi9 * 0.3 + rsi14 * 0.4 + rsi21 * 0.3 - 50) / 50;
            output.AddFeature("04_tech_rsi_composite", rsiComposite);

            // ===== 2. MACD-BASED FEATURES (TRANSFORMED) =====

            var (macdLine, macdSignal, macdHist) = CalculateMACD(bars, currentIndex);
            _macdHistory.Add(macdHist);

            // MACD histogram normalized by ATR for comparability
            var atr = CalculateATR(bars, currentIndex, 14);
            var macdNormalized = SafeDiv(macdHist, atr);
            output.AddFeature("04_tech_macd_normalized", Sigmoid(macdNormalized));

            // MACD momentum quality
            var macdQuality = CalculateMACDQuality(_macdHistory);
            output.AddFeature("04_tech_macd_quality", macdQuality);

            // ===== 4. BOLLINGER BANDS FEATURES (TRANSFORMED) =====

            var (bbUpper, bbMiddle, bbLower, bbWidth) = CalculateBollingerBands(bars, currentIndex, 20, 2);

            // BB position as directional signal
            var bbPosition = SafeDiv(close - bbLower, bbUpper - bbLower) * 2 - 1;  // [-1, 1]
            output.AddFeature("04_tech_bb_position", bbPosition);

            // BB squeeze detection (volatility contraction)
            var bbSqueeze = DetectBBSqueeze(bars, currentIndex, bbWidth);
            output.AddFeature("04_tech_bb_squeeze", bbSqueeze);

            // BB expansion signal (breakout potential)
            var bbExpansion = CalculateBBExpansion(bars, currentIndex, bbWidth);
            output.AddFeature("04_tech_bb_expansion", bbExpansion);

            // ===== 5. MOVING AVERAGE FEATURES (TRANSFORMED) =====

            var ema9 = CalculateEMA(bars, currentIndex, 9);
            var ema21 = CalculateEMA(bars, currentIndex, 21);
            var ema50 = CalculateEMA(bars, currentIndex, 50);

            // MA alignment signal (trend confirmation)
            var maAlignment = CalculateMAAlignment(close, ema9, ema21, ema50);
            output.AddFeature("04_tech_ma_alignment", maAlignment);

            // Price-MA deviation (mean reversion)
            var maDev9 = SafeDiv(close - ema9, atr);
            output.AddFeature("04_tech_ma_dev_9", Sigmoid(maDev9));

            var maDev21 = SafeDiv(close - ema21, atr);
            output.AddFeature("04_tech_ma_dev_21", Sigmoid(maDev21));

            // MA slope convergence/divergence
            var maConvergence = CalculateMAConvergence(bars, currentIndex);
            output.AddFeature("04_tech_ma_convergence", maConvergence);

            // ===== 6. ATR/VOLATILITY FEATURES (TRANSFORMED) =====

            var atr14 = CalculateATR(bars, currentIndex, 14);
            var atr7 = CalculateATR(bars, currentIndex, 7);

            // ATR expansion ratio (volatility regime)
            var atrRatio = SafeDiv(atr7, atr14);
            output.AddFeature("04_tech_atr_ratio", atrRatio);

            // Volatility percentile
            var volPercentile = CalculateVolatilityPercentile(bars, currentIndex, atr14);
            output.AddFeature("04_tech_vol_percentile", volPercentile);
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

        private double CalculateMACDQuality(RollingWindow<double> history)
        {
            if (history.Count < 5) return 0;

            var recent = history.GetValues().Take(5).ToList();
            return CalculateMomentumQuality(recent);
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

        public override void Reset()
        {
            _rsiHistory.Clear();
            _macdHistory.Clear();
        }
    }
}