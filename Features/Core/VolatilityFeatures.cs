using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Volatility features for risk assessment and regime detection
    /// </summary>
    public class VolatilityFeatures : BaseCalculator
    {
        public override string Name => "Volatility";
        public override string Category => "Core";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 3;

        private readonly RollingWindow<double> _atrValues = new(50);
        private readonly RollingWindow<double> _realized = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            var close = (double)bars[currentIndex].Close;

            // === 1. Normalized ATR ===
            var atr14 = ATR(bars, 14, currentIndex);
            _atrValues.Add(atr14);

            // ATR as percentage of price
            var atrPercent = SafeDiv(atr14, close) * 100;
            output.AddFeature("atr_percent", Math.Min(atrPercent, 5)); // Cap at 5%

            // === 2. ATR Regime ===
            if (_atrValues.Count >= 20)
            {
                var atrMean = _atrValues.GetValues().Take(20).Average();
                var atrStd = Math.Sqrt(_atrValues.GetValues().Take(20).Select(a => Math.Pow(a - atrMean, 2)).Average());

                // Volatility regime: low, normal, high
                var volRegime = 0.0;
                if (atr14 > atrMean + atrStd) volRegime = 1.0; // High volatility
                else if (atr14 < atrMean - atrStd) volRegime = -1.0; // Low volatility

                output.AddFeature("volatility_regime", volRegime);

                // Volatility expansion
                var volExpansion = SafeDiv(atr14 - atrMean, atrMean);
                output.AddFeature("vol_expansion", Sigmoid(volExpansion * 10));
            }

            // === 3. Bollinger Band Features ===
            if (currentIndex >= 20)
            {
                var sma20 = SMA(bars, 20, currentIndex);
                var stdDev = StdDev(bars, 20, currentIndex);

                var bbUpper = sma20 + 2 * stdDev;
                var bbLower = sma20 - 2 * stdDev;
                var bbWidth = bbUpper - bbLower;

                // Position within bands (0-1 scale)
                var bbPosition = SafeDiv(close - bbLower, bbWidth);
                output.AddFeature("bb_position", Math.Max(0, Math.Min(1, bbPosition)));

                // Band squeeze (low volatility setup)
                var bbSqueeze = SafeDiv(bbWidth, sma20) * 100;
                output.AddFeature("bb_squeeze", Math.Min(bbSqueeze, 10));

                // Distance from bands (potential reversal)
                var distToUpper = SafeDiv(bbUpper - close, stdDev);
                var distToLower = SafeDiv(close - bbLower, stdDev);
                output.AddFeature("bb_dist_upper", Math.Min(distToUpper, 3));
                output.AddFeature("bb_dist_lower", Math.Min(distToLower, 3));
            }

            // === 4. Realized Volatility ===
            if (currentIndex >= 20)
            {
                var returns = new List<double>();
                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    if (i > 0)
                    {
                        var ret = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                        returns.Add(ret);
                    }
                }

                var realizedVol = Math.Sqrt(returns.Select(r => r * r).Sum()) * Math.Sqrt(252 * 24 * 60); // Annualized
                _realized.Add(realizedVol);

                // Normalized realized volatility
                if (_realized.Count >= 20)
                {
                    var rvMean = _realized.GetValues().Take(20).Average();
                    var rvStd = Math.Sqrt(_realized.GetValues().Take(20).Select(r => Math.Pow(r - rvMean, 2)).Average());

                    var rvZScore = AdaptiveZScore(realizedVol, rvMean, rvStd);
                    output.AddFeature("realized_vol_zscore", rvZScore);
                }
            }

            // === 5. Volatility Clustering ===
            // GARCH-like effect: high vol follows high vol
            if (_atrValues.Count >= 10)
            {
                var recent = _atrValues.GetValues().Take(5).Average();
                var older = _atrValues.GetValues().Skip(5).Take(5).Average();

                var volClustering = SafeDiv(recent - older, older);
                output.AddFeature("vol_clustering", Sigmoid(volClustering * 10));
            }

            // === 6. Range Metrics ===
            var range = (double)(bars[currentIndex].High - bars[currentIndex].Low);

            // Average range over different periods
            if (currentIndex >= 10)
            {
                var avgRange10 = 0.0;
                for (int i = currentIndex - 9; i <= currentIndex; i++)
                {
                    avgRange10 += (double)(bars[i].High - bars[i].Low);
                }
                avgRange10 /= 10;

                var rangeRatio = SafeDiv(range, avgRange10);
                output.AddFeature("range_expansion", Sigmoid((rangeRatio - 1) * 5));
            }

            // === 7. Parkinson Volatility ===
            // High-Low based volatility estimator
            if (currentIndex >= 20)
            {
                var parkinsonSum = 0.0;
                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    var hl = Math.Log((double)bars[i].High / (double)bars[i].Low);
                    parkinsonSum += hl * hl;
                }

                var parkinson = Math.Sqrt(parkinsonSum / (20 * 4 * Math.Log(2))) * Math.Sqrt(252 * 24 * 60);
                output.AddFeature("parkinson_vol", Math.Min(parkinson, 1));
            }
        }

        public override void Reset()
        {
            base.Reset();
            _atrValues.Clear();
            _realized.Clear();
        }
    }
}