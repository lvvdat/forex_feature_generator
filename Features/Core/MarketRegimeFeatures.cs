using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ForexFeatureGenerator.Features.ML
{
    /// <summary>
    /// Advanced market regime detection features
    /// </summary>
    public class MarketRegimeFeatures : BaseCalculator
    {
        public override string Name => "MarketRegime";
        public override string Category => "ML";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 15;

        private readonly RollingWindow<double> _regimeScores = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return;

            // === 1. Trend Regime ===
            var trendScore = CalculateTrendRegime(bars, currentIndex);
            output.AddFeature("trend_regime", trendScore);

            // === 2. Volatility Regime ===
            var volRegime = CalculateVolatilityRegime(bars, currentIndex);
            output.AddFeature("vol_regime_score", volRegime);

            // === 3. Market Efficiency ===
            var efficiency = CalculateMarketEfficiency(bars, currentIndex);
            output.AddFeature("market_efficiency", efficiency);

            // === 4. Regime Transition ===
            _regimeScores.Add(trendScore);

            if (_regimeScores.Count >= 10)
            {
                var recent = _regimeScores.GetValues().Take(5).Average();
                var older = _regimeScores.GetValues().Skip(5).Take(5).Average();

                var transition = recent - older;
                output.AddFeature("regime_transition", Sigmoid(transition * 10));
            }

            // === 5. Hurst Exponent ===
            // Measures trending vs mean-reverting behavior
            var hurst = CalculateHurstExponent(bars, currentIndex);
            output.AddFeature("hurst_exponent", hurst);

            // === 6. Market State ===
            // Composite market state indicator
            var marketState = DetermineMarketState(trendScore, volRegime, efficiency, hurst);
            output.AddFeature("market_state", marketState);
        }

        private double CalculateTrendRegime(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // ADX-based trend strength
            var plusDM = 0.0;
            var minusDM = 0.0;
            var tr = 0.0;

            for (int i = currentIndex - 13; i <= currentIndex; i++)
            {
                if (i <= 0) continue;

                var highDiff = (double)(bars[i].High - bars[i - 1].High);
                var lowDiff = (double)(bars[i - 1].Low - bars[i].Low);

                if (highDiff > lowDiff && highDiff > 0)
                    plusDM += highDiff;
                if (lowDiff > highDiff && lowDiff > 0)
                    minusDM += lowDiff;

                tr += TrueRange(bars, i);
            }

            if (tr == 0) return 0;

            var plusDI = plusDM / tr;
            var minusDI = minusDM / tr;
            var adx = SafeDiv(Math.Abs(plusDI - minusDI), plusDI + minusDI);

            // Directional trend score
            var direction = plusDI > minusDI ? 1.0 : -1.0;
            return direction * adx;
        }

        private double CalculateVolatilityRegime(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // GARCH-style volatility clustering
            var returns = new List<double>();

            for (int i = currentIndex - 29; i <= currentIndex; i++)
            {
                if (i > 0)
                {
                    var ret = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                    returns.Add(ret);
                }
            }

            if (returns.Count < 20) return 0.5;

            // Recent vs historical volatility
            var recentVol = Math.Sqrt(returns.Take(10).Select(r => r * r).Average());
            var historicalVol = Math.Sqrt(returns.Skip(10).Select(r => r * r).Average());

            var volRatio = SafeDiv(recentVol, historicalVol);

            // Classify regime
            if (volRatio > 1.5) return 1.0; // High volatility
            if (volRatio < 0.7) return -1.0; // Low volatility
            return 0.0; // Normal
        }

        private double CalculateMarketEfficiency(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Fractal Efficiency (Kaufman)
            if (currentIndex < 20) return 0.5;

            var netChange = Math.Abs((double)(bars[currentIndex].Close - bars[currentIndex - 19].Close));
            var sumMovements = 0.0;

            for (int i = currentIndex - 18; i <= currentIndex; i++)
            {
                sumMovements += Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
            }

            return SafeDiv(netChange, sumMovements, 0.5);
        }

        private double CalculateHurstExponent(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified R/S analysis
            if (currentIndex < 50) return 0.5;

            var returns = new List<double>();
            for (int i = currentIndex - 49; i <= currentIndex; i++)
            {
                if (i > 0)
                {
                    returns.Add(Math.Log((double)bars[i].Close / (double)bars[i - 1].Close));
                }
            }

            var mean = returns.Average();
            var cumDev = new List<double>();
            var cumSum = 0.0;

            foreach (var ret in returns)
            {
                cumSum += ret - mean;
                cumDev.Add(cumSum);
            }

            var range = cumDev.Max() - cumDev.Min();
            var std = Math.Sqrt(returns.Select(r => Math.Pow(r - mean, 2)).Average());

            if (std == 0) return 0.5;

            var rs = range / std;
            var hurst = Math.Log(rs) / Math.Log(returns.Count);

            return Math.Max(0, Math.Min(1, hurst));
        }

        private double DetermineMarketState(double trend, double vol, double efficiency, double hurst)
        {
            // Composite market state score
            // Trending: high trend, high efficiency, hurst > 0.5
            // Ranging: low trend, low efficiency, hurst ~= 0.5
            // Volatile: high vol, low efficiency

            var trendingScore = trend * efficiency * (hurst - 0.5) * 2;
            var rangingScore = (1 - Math.Abs(trend)) * (1 - efficiency);
            var volatileScore = vol * (1 - efficiency);

            // Find dominant state
            if (trendingScore > rangingScore && trendingScore > volatileScore)
                return Math.Sign(trend); // 1 for uptrend, -1 for downtrend
            else if (volatileScore > rangingScore)
                return 0.5; // Volatile
            else
                return 0; // Ranging
        }

        public override void Reset()
        {
            base.Reset();
            _regimeScores.Clear();
        }
    }
}