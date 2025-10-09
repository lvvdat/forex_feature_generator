using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Market microstructure features for order flow and liquidity
    /// </summary>
    public class MicrostructureFeatures : BaseCalculator
    {
        public override string Name => "Microstructure";
        public override string Category => "Core";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 4;

        private readonly RollingWindow<double> _volumeProfile = new(50);
        private readonly RollingWindow<double> _orderFlow = new(20);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // === 1. Order Flow Imbalance ===
            var buyVolume = bar.UpVolume;
            var sellVolume = bar.DownVolume;
            var totalVolume = bar.TickVolume;

            if (totalVolume > 0)
            {
                // Normalized order flow (-1 to 1)
                var orderFlowImbalance = SafeDiv(buyVolume - sellVolume, totalVolume);
                output.AddFeature("order_flow_imbalance", orderFlowImbalance);

                _orderFlow.Add(orderFlowImbalance);

                // Cumulative order flow
                if (_orderFlow.Count >= 10)
                {
                    var cumFlow = _orderFlow.GetValues().Take(10).Sum();
                    output.AddFeature("cumulative_order_flow", Sigmoid(cumFlow));

                    // Order flow momentum
                    var flowMomentum = _orderFlow[0] - _orderFlow[9];
                    output.AddFeature("order_flow_momentum", flowMomentum);
                }
            }

            // === 2. Volume Analysis ===
            _volumeProfile.Add(totalVolume);

            if (_volumeProfile.Count >= 20)
            {
                var volumes = _volumeProfile.GetValues().Take(20).ToArray();
                var avgVolume = volumes.Average();
                var volStd = Math.Sqrt(volumes.Select(v => Math.Pow(v - avgVolume, 2)).Average());

                // Volume surge detection
                var volumeZScore = volStd > 0 ? (totalVolume - avgVolume) / volStd : 0;
                output.AddFeature("volume_surge", Math.Min(Math.Max(volumeZScore, -3), 3));

                // Volume trend
                if (_volumeProfile.Count >= 10)
                {
                    var recentVol = _volumeProfile.GetValues().Take(5).Average();
                    var olderVol = _volumeProfile.GetValues().Skip(5).Take(5).Average();
                    var volTrend = SafeDiv(recentVol - olderVol, olderVol);
                    output.AddFeature("volume_trend", Sigmoid(volTrend * 10));
                }
            }

            // === 3. Spread Analysis ===
            var spread = (double)bar.AvgSpread;
            var spreadBps = SafeDiv(spread, close) * 10000; // Basis points

            // Adaptive spread normalization
            if (currentIndex >= 20)
            {
                var spreads = new List<double>();
                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    spreads.Add(SafeDiv((double)bars[i].AvgSpread, (double)bars[i].Close) * 10000);
                }

                var avgSpread = spreads.Average();
                var spreadStd = Math.Sqrt(spreads.Select(s => Math.Pow(s - avgSpread, 2)).Average());

                var spreadZScore = AdaptiveZScore(spreadBps, avgSpread, spreadStd);
                output.AddFeature("spread_zscore", spreadZScore);

                // Spread widening (liquidity stress)
                var spreadWidening = spreadBps > avgSpread + spreadStd ? 1.0 : 0.0;
                output.AddFeature("spread_widening", spreadWidening);
            }

            // === 4. Price Impact ===
            // How much does volume move price
            if (totalVolume > 0 && currentIndex >= 1)
            {
                var priceMove = Math.Abs(close - (double)bars[currentIndex - 1].Close);
                var priceImpact = SafeDiv(priceMove, totalVolume) * 1000000;

                // Normalized price impact
                if (currentIndex >= 20)
                {
                    var impacts = new List<double>();
                    for (int i = currentIndex - 19; i <= currentIndex; i++)
                    {
                        if (i > 0 && bars[i].TickVolume > 0)
                        {
                            var pm = Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
                            impacts.Add(SafeDiv(pm, bars[i].TickVolume) * 1000000);
                        }
                    }

                    if (impacts.Count > 0)
                    {
                        var avgImpact = impacts.Average();
                        var impactRatio = SafeDiv(priceImpact, avgImpact);
                        output.AddFeature("price_impact_ratio", Math.Min(impactRatio, 5));
                    }
                }
            }

            // === 5. VWAP Deviation ===
            if (currentIndex >= 20)
            {
                double vwapNum = 0;
                double vwapDen = 0;

                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    var typical = (double)bars[i].TypicalPrice;
                    var vol = bars[i].TickVolume;
                    vwapNum += typical * vol;
                    vwapDen += vol;
                }

                if (vwapDen > 0)
                {
                    var vwap = vwapNum / vwapDen;
                    var vwapDev = SafeDiv(close - vwap, vwap) * 10000;

                    // Bounded VWAP deviation
                    output.AddFeature("vwap_deviation", Math.Max(-100, Math.Min(100, vwapDev)));

                    // VWAP trend
                    if (currentIndex >= 30)
                    {
                        double vwapNum2 = 0;
                        double vwapDen2 = 0;

                        for (int i = currentIndex - 29; i <= currentIndex - 10; i++)
                        {
                            var typical = (double)bars[i].TypicalPrice;
                            var vol = bars[i].TickVolume;
                            vwapNum2 += typical * vol;
                            vwapDen2 += vol;
                        }

                        if (vwapDen2 > 0)
                        {
                            var vwap2 = vwapNum2 / vwapDen2;
                            var vwapSlope = SafeDiv(vwap - vwap2, vwap2) * 10000;
                            output.AddFeature("vwap_slope", Sigmoid(vwapSlope));
                        }
                    }
                }
            }

            // === 6. Tick Distribution ===
            // How evenly distributed are ticks (market activity)
            if (totalVolume > 0)
            {
                var buyRatio = SafeDiv(buyVolume, totalVolume);
                var sellRatio = SafeDiv(sellVolume, totalVolume);

                // Entropy-like measure (-1 = all one side, 0 = balanced)
                var tickEntropy = 0.0;
                if (buyRatio > 0 && sellRatio > 0)
                {
                    tickEntropy = -(buyRatio * Math.Log(buyRatio) + sellRatio * Math.Log(sellRatio)) / Math.Log(2);
                }
                output.AddFeature("tick_distribution", tickEntropy);
            }
        }

        public override void Reset()
        {
            base.Reset();
            _volumeProfile.Clear();
            _orderFlow.Clear();
        }
    }
}