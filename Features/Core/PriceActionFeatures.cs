using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Core price action features optimized for classification
    /// Focuses on relative movements and patterns
    /// </summary>
    public class PriceActionFeatures : BaseCalculator
    {
        public override string Name => "PriceAction";
        public override string Category => "Core";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 1;

        private readonly RollingWindow<double> _priceChanges = new(50);
        private readonly RollingWindow<double> _ranges = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;
            var open = (double)bar.Open;
            var high = (double)bar.High;
            var low = (double)bar.Low;
            var range = high - low;

            // === 1. Normalized Price Position ===
            // Where is price within recent range (0-1 scale)
            var highest20 = double.MinValue;
            var lowest20 = double.MaxValue;
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                highest20 = Math.Max(highest20, (double)bars[i].High);
                lowest20 = Math.Min(lowest20, (double)bars[i].Low);
            }
            var pricePosition = SafeDiv(close - lowest20, highest20 - lowest20);
            output.AddFeature("price_position_20", pricePosition);

            // === 2. Directional Movement ===
            // Smoothed directional change
            if (currentIndex >= 1)
            {
                var change = close - (double)bars[currentIndex - 1].Close;
                _priceChanges.Add(change);

                if (_priceChanges.Count >= 5)
                {
                    var recentChanges = _priceChanges.GetValues().Take(5).ToArray();
                    var avgChange = recentChanges.Average();
                    var changeStd = Math.Sqrt(recentChanges.Select(c => Math.Pow(c - avgChange, 2)).Average());

                    // Normalized directional strength
                    var dirStrength = changeStd > 0 ? avgChange / changeStd : 0;
                    output.AddFeature("directional_strength", Sigmoid(dirStrength, 0.5));
                }
            }

            // === 3. Volatility-Adjusted Return ===
            if (currentIndex >= 10)
            {
                var return10 = SafeDiv(close - (double)bars[currentIndex - 10].Close, (double)bars[currentIndex - 10].Close);
                var atr = ATR(bars, 10, currentIndex);
                var volAdjReturn = atr > 0 ? return10 / atr : 0;
                output.AddFeature("vol_adj_return_10", Sigmoid(volAdjReturn * 100));
            }

            // === 4. Candle Patterns (Simplified) ===
            var bodySize = Math.Abs(close - open);
            var upperWick = high - Math.Max(open, close);
            var lowerWick = Math.Min(open, close) - low;

            if (range > 0)
            {
                // Body dominance (large body = trend)
                output.AddFeature("body_dominance", SafeDiv(bodySize, range));

                // Wick imbalance (buying/selling pressure)
                var wickImbalance = SafeDiv(upperWick - lowerWick, range);
                output.AddFeature("wick_imbalance", wickImbalance);

                // Pin bar detection
                var isPinBar = (upperWick > bodySize * 2 || lowerWick > bodySize * 2) ? 1.0 : 0.0;
                output.AddFeature("pin_bar", isPinBar);
            }

            // === 5. Support/Resistance Distance ===
            // Distance from recent highs/lows
            if (currentIndex >= 50)
            {
                var recentHigh = double.MinValue;
                var recentLow = double.MaxValue;

                for (int i = currentIndex - 49; i <= currentIndex - 1; i++)
                {
                    recentHigh = Math.Max(recentHigh, (double)bars[i].High);
                    recentLow = Math.Min(recentLow, (double)bars[i].Low);
                }

                var distToResistance = SafeDiv(recentHigh - close, close);
                var distToSupport = SafeDiv(close - recentLow, close);

                output.AddFeature("dist_to_resistance", Math.Min(distToResistance * 10000, 100));
                output.AddFeature("dist_to_support", Math.Min(distToSupport * 10000, 100));
            }

            // === 6. Price Efficiency ===
            // How directly price moves (trend quality)
            if (currentIndex >= 20)
            {
                var netMove = Math.Abs(close - (double)bars[currentIndex - 19].Close);
                var pathLength = 0.0;

                for (int i = currentIndex - 18; i <= currentIndex; i++)
                {
                    pathLength += Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
                }

                var efficiency = SafeDiv(netMove, pathLength);
                output.AddFeature("price_efficiency", efficiency);
            }

            // === 7. Volume-Price Relationship ===
            if (bar.TickVolume > 0)
            {
                _ranges.Add(range);

                if (_ranges.Count >= 20)
                {
                    var avgRange = _ranges.GetValues().Take(20).Average();
                    var rangeRatio = SafeDiv(range, avgRange);

                    var avgVolume = 0.0;
                    for (int i = currentIndex - 19; i <= currentIndex; i++)
                    {
                        avgVolume += bars[i].TickVolume;
                    }
                    avgVolume /= 20;

                    var volumeRatio = SafeDiv(bar.TickVolume, avgVolume);

                    // High volume + small range = accumulation/distribution
                    // High volume + large range = breakout
                    var volPriceSignal = volumeRatio * (2 - rangeRatio);
                    output.AddFeature("volume_price_signal", Math.Min(volPriceSignal, 3));
                }
            }

            // === 8. Momentum Quality ===
            // Consistency of price movement
            if (_priceChanges.Count >= 10)
            {
                var changes = _priceChanges.GetValues().Take(10).ToArray();
                var positiveCount = changes.Count(c => c > 0);
                var momentumQuality = (positiveCount - 5) / 5.0; // -1 to 1 scale
                output.AddFeature("momentum_quality", momentumQuality);
            }
        }

        public override void Reset()
        {
            base.Reset();
            _priceChanges.Clear();
            _ranges.Clear();
        }
    }
}