using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Momentum indicators optimized for trend detection
    /// </summary>
    public class MomentumFeatures : BaseCalculator
    {
        public override string Name => "Momentum";
        public override string Category => "Core";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 2;

        private readonly RollingWindow<double> _rsiValues = new(20);
        private readonly RollingWindow<double> _macdHistogram = new(20);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 26) return;

            var close = (double)bars[currentIndex].Close;

            // === 1. RSI with Divergence ===
            var rsi = RSI(bars, 14, currentIndex);
            _rsiValues.Add(rsi);

            // Transform RSI to -1 to 1 scale for classification
            var rsiNormalized = (rsi - 50) / 50;
            output.AddFeature("rsi_normalized", rsiNormalized);

            // RSI momentum (change in RSI)
            if (_rsiValues.Count >= 5)
            {
                var rsiMomentum = _rsiValues[0] - _rsiValues[4];
                output.AddFeature("rsi_momentum", Sigmoid(rsiMomentum / 10));
            }

            // === 2. MACD Signal ===
            var ema12 = EMA(bars, 12, currentIndex);
            var ema26 = EMA(bars, 26, currentIndex);
            var macdLine = ema12 - ema26;

            if (currentIndex >= 35)
            {
                // Calculate signal line (9-period EMA of MACD)
                var macdValues = new List<double>();
                for (int i = currentIndex - 8; i <= currentIndex; i++)
                {
                    var e12 = EMA(bars, 12, i);
                    var e26 = EMA(bars, 26, i);
                    macdValues.Add(e12 - e26);
                }
                var signal = macdValues.Average();
                var histogram = macdLine - signal;

                _macdHistogram.Add(histogram);

                // Normalized MACD histogram
                if (_macdHistogram.Count >= 20)
                {
                    var histValues = _macdHistogram.GetValues().Take(20).ToArray();
                    var histMean = histValues.Average();
                    var histStd = Math.Sqrt(histValues.Select(h => Math.Pow(h - histMean, 2)).Average());

                    var macdZScore = AdaptiveZScore(histogram, histMean, histStd);
                    output.AddFeature("macd_zscore", macdZScore);

                    // MACD histogram slope (acceleration)
                    if (_macdHistogram.Count >= 5)
                    {
                        var recentHist = _macdHistogram.GetValues().Take(5).ToArray();
                        var histSlope = CalculateSlope(recentHist);
                        output.AddFeature("macd_acceleration", Sigmoid(histSlope * 1000));
                    }
                }
            }

            // === 3. Rate of Change (ROC) ===
            if (currentIndex >= 10)
            {
                var roc10 = SafeDiv(close - (double)bars[currentIndex - 10].Close, (double)bars[currentIndex - 10].Close) * 100;

                // Adaptive ROC normalization
                var rocValues = new List<double>();
                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    if (i >= 10)
                    {
                        var r = SafeDiv((double)(bars[i].Close - bars[i - 10].Close), (double)bars[i - 10].Close) * 100;
                        rocValues.Add(r);
                    }
                }

                if (rocValues.Count > 0)
                {
                    var rocMean = rocValues.Average();
                    var rocStd = Math.Sqrt(rocValues.Select(r => Math.Pow(r - rocMean, 2)).Average());
                    var rocZScore = AdaptiveZScore(roc10, rocMean, rocStd);
                    output.AddFeature("roc_adaptive", rocZScore);
                }
            }

            // === 4. Momentum Oscillator ===
            // Custom momentum based on multiple timeframes
            if (currentIndex >= 20)
            {
                var mom5 = close - (double)bars[currentIndex - 5].Close;
                var mom10 = close - (double)bars[currentIndex - 10].Close;
                var mom20 = close - (double)bars[currentIndex - 20].Close;

                // Weighted momentum (recent changes more important)
                var weightedMom = (mom5 * 0.5 + mom10 * 0.3 + mom20 * 0.2) / close * 10000;
                output.AddFeature("weighted_momentum", Sigmoid(weightedMom));

                // Momentum alignment (all timeframes agree)
                var momAlignment = 0.0;
                if (mom5 > 0 && mom10 > 0 && mom20 > 0) momAlignment = 1.0;
                else if (mom5 < 0 && mom10 < 0 && mom20 < 0) momAlignment = -1.0;
                output.AddFeature("momentum_alignment", momAlignment);
            }

            // === 5. Stochastic RSI ===
            if (_rsiValues.Count >= 14)
            {
                var rsiWindow = _rsiValues.GetValues().Take(14).ToArray();
                var rsiMin = rsiWindow.Min();
                var rsiMax = rsiWindow.Max();

                if (rsiMax - rsiMin > 0)
                {
                    var stochRsi = SafeDiv(rsi - rsiMin, rsiMax - rsiMin);
                    output.AddFeature("stoch_rsi", stochRsi);
                }
            }

            // === 6. Price Acceleration ===
            if (currentIndex >= 3)
            {
                var vel1 = (double)(bars[currentIndex].Close - bars[currentIndex - 1].Close);
                var vel2 = (double)(bars[currentIndex - 1].Close - bars[currentIndex - 2].Close);
                var acceleration = (vel1 - vel2) / close * 10000;
                output.AddFeature("price_acceleration", Sigmoid(acceleration, 2));
            }

            // === 7. Momentum Divergence ===
            // Price vs RSI divergence
            if (currentIndex >= 20 && _rsiValues.Count >= 20)
            {
                // Price trend
                var priceTrend = close > (double)bars[currentIndex - 10].Close ? 1 : -1;

                // RSI trend
                var rsiTrend = _rsiValues[0] > _rsiValues[10] ? 1 : -1;

                // Divergence: price up but RSI down (bearish) or price down but RSI up (bullish)
                var divergence = 0.0;
                if (priceTrend > 0 && rsiTrend < 0) divergence = -1.0; // Bearish divergence
                else if (priceTrend < 0 && rsiTrend > 0) divergence = 1.0; // Bullish divergence

                output.AddFeature("momentum_divergence", divergence);
            }
        }

        public override void Reset()
        {
            base.Reset();
            _rsiValues.Clear();
            _macdHistogram.Clear();
        }
    }
}