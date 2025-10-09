using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ForexFeatureGenerator.Features.Cross
{
    /// <summary>
    /// Cross-timeframe features for multi-scale analysis
    /// Requires both M1 and M5 aggregators
    /// </summary>
    public class CrossTimeframeFeatures : BaseCalculator
    {
        public override string Name => "CrossTimeframe";
        public override string Category => "Cross";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5); // Primary timeframe
        public override int Priority => 10;

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // === 1. Trend Alignment ===
            // Check if short and long-term trends align
            var ema9 = EMA(bars, 9, currentIndex);
            var ema21 = EMA(bars, 21, currentIndex);
            var ema50 = EMA(bars, 50, currentIndex);

            // Perfect alignment: 1 = bullish, -1 = bearish, 0 = mixed
            var trendAlignment = 0.0;
            if (close > ema9 && ema9 > ema21 && ema21 > ema50)
                trendAlignment = 1.0;
            else if (close < ema9 && ema9 < ema21 && ema21 < ema50)
                trendAlignment = -1.0;

            output.AddFeature("trend_alignment", trendAlignment);

            // === 2. Momentum Divergence ===
            // Compare momentum across timeframes
            if (currentIndex >= 10)
            {
                var shortMomentum = close - (double)bars[currentIndex - 2].Close;
                var longMomentum = close - (double)bars[currentIndex - 10].Close;

                // Normalized divergence
                var momDivergence = 0.0;
                if (shortMomentum > 0 && longMomentum < 0)
                    momDivergence = -1.0; // Short-term rally in downtrend
                else if (shortMomentum < 0 && longMomentum > 0)
                    momDivergence = 1.0; // Short-term pullback in uptrend

                output.AddFeature("momentum_divergence_tf", momDivergence);
            }

            // === 3. Volatility Ratio ===
            // Compare short vs long-term volatility
            var atr10 = ATR(bars, 10, currentIndex);
            var atr30 = ATR(bars, 30, currentIndex);

            var volRatio = SafeDiv(atr10, atr30);
            output.AddFeature("volatility_ratio_tf", Sigmoid((volRatio - 1) * 5));

            // === 4. Support/Resistance Confluence ===
            // Multiple timeframe S/R levels
            var levels = new List<double>();

            // Recent highs/lows at different scales
            for (int period = 10; period <= 50; period += 10)
            {
                if (currentIndex >= period)
                {
                    var high = double.MinValue;
                    var low = double.MaxValue;

                    for (int i = currentIndex - period + 1; i <= currentIndex; i++)
                    {
                        high = Math.Max(high, (double)bars[i].High);
                        low = Math.Min(low, (double)bars[i].Low);
                    }

                    levels.Add(high);
                    levels.Add(low);
                }
            }

            // Find nearest level
            if (levels.Count > 0)
            {
                var nearestLevel = levels.OrderBy(l => Math.Abs(l - close)).First();
                var distToLevel = SafeDiv(Math.Abs(close - nearestLevel), close) * 10000;
                output.AddFeature("nearest_level_distance", Math.Min(distToLevel, 50));

                // Level strength (how many timeframes agree)
                var levelStrength = levels.Count(l => Math.Abs(l - nearestLevel) / nearestLevel < 0.001) / (double)levels.Count;
                output.AddFeature("level_confluence", levelStrength);
            }

            // === 5. Volume Profile Divergence ===
            // Compare volume patterns across timeframes
            if (currentIndex >= 20)
            {
                var shortVol = 0;
                var longVol = 0;

                for (int i = currentIndex - 4; i <= currentIndex; i++)
                    shortVol += bars[i].TickVolume;

                for (int i = currentIndex - 19; i <= currentIndex - 15; i++)
                    longVol += bars[i].TickVolume;

                var volDivergence = SafeDiv(shortVol - longVol, longVol);
                output.AddFeature("volume_divergence_tf", Sigmoid(volDivergence * 5));
            }

            // === 6. Fractal Dimension ===
            // Market complexity/trendiness
            var fractalDim = CalculateFractalDimension(bars, currentIndex, 30);
            output.AddFeature("fractal_dimension", fractalDim);

            // === 7. Multi-Scale RSI ===
            // RSI agreement across timeframes
            if (currentIndex >= 21)
            {
                var rsi10 = RSI(bars, 10, currentIndex);
                var rsi14 = RSI(bars, 14, currentIndex);
                var rsi21 = RSI(bars, 21, currentIndex);

                // Average RSI (robust measure)
                var avgRsi = (rsi10 + rsi14 + rsi21) / 3;
                var rsiConvergence = 1 - (Math.Abs(rsi10 - avgRsi) + Math.Abs(rsi14 - avgRsi) + Math.Abs(rsi21 - avgRsi)) / 150;

                output.AddFeature("rsi_convergence", rsiConvergence);
                output.AddFeature("rsi_composite", (avgRsi - 50) / 50);
            }

            // === 8. Timeframe Momentum Score ===
            // Composite momentum across scales
            if (currentIndex >= 30)
            {
                var scores = new List<double>();

                foreach (int period in new[] { 5, 10, 20, 30 })
                {
                    var ret = SafeDiv(close - (double)bars[currentIndex - period + 1].Close,
                                     (double)bars[currentIndex - period + 1].Close);
                    scores.Add(ret);
                }

                // Weighted average (recent more important)
                var weightedScore = scores[0] * 0.4 + scores[1] * 0.3 + scores[2] * 0.2 + scores[3] * 0.1;
                output.AddFeature("tf_momentum_score", Sigmoid(weightedScore * 1000));
            }
        }

        private double CalculateFractalDimension(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            if (currentIndex < period) return 1.5;

            var prices = new List<double>();
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                prices.Add((double)bars[i].Close);
            }

            var maxPrice = prices.Max();
            var minPrice = prices.Min();
            var range = maxPrice - minPrice;

            if (range < 1e-10) return 1.5;

            // Simplified box-counting
            int[] boxSizes = { 2, 4, 8 };
            var boxCounts = new List<double>();

            foreach (var size in boxSizes)
            {
                var boxes = new HashSet<int>();
                for (int i = 0; i < prices.Count; i++)
                {
                    var box = (int)((prices[i] - minPrice) / range * size);
                    boxes.Add(box);
                }
                boxCounts.Add(boxes.Count);
            }

            // Log-log regression for fractal dimension
            if (boxCounts.Count >= 2)
            {
                var logSizes = boxSizes.Select(s => Math.Log(s)).ToArray();
                var logCounts = boxCounts.Select(c => Math.Log(c)).ToArray();

                var slope = CalculateSlope(logCounts);
                return Math.Min(2, Math.Max(1, Math.Abs(slope)));
            }

            return 1.5;
        }

        public override void Reset()
        {
            base.Reset();
        }
    }
}