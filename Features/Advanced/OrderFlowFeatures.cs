using System.Collections.Concurrent;

using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Advanced
{
    public class OrderFlowFeatures : BaseFeatureCalculator
    {
        public override string Name => "OrderFlow";
        public override string Category => "Microstructure";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 10;

        // Order flow tracking
        private readonly RollingWindow<OrderFlowSnapshot> _flowHistory = new(100);
        private readonly ConcurrentDictionary<decimal, int> _bidLevels = new();
        private readonly ConcurrentDictionary<decimal, int> _askLevels = new();

        public class OrderFlowSnapshot
        {
            public DateTime Timestamp { get; set; }
            public decimal TotalBuyVolume { get; set; }
            public decimal TotalSellVolume { get; set; }
            public decimal NetOrderFlow { get; set; }
            public decimal CumulativeDelta { get; set; }
            public int BuyTradeCount { get; set; }
            public int SellTradeCount { get; set; }
            public decimal VWAP { get; set; }
            public decimal PressureRatio { get; set; }
            public decimal LargeOrderRatio { get; set; }
        }

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return;

            var bar = bars[currentIndex];

            // Calculate order flow metrics
            var buyVolume = bar.UpVolume;
            var sellVolume = bar.DownVolume;
            var totalVolume = buyVolume + sellVolume;

            if (totalVolume == 0) return;

            // Net order flow
            var netFlow = buyVolume - sellVolume;
            output.AddFeature("fg2_of_net_flow", (double)netFlow);

            // Cumulative delta (last 20 bars)
            double cumulativeDelta = 0;
            if (currentIndex >= 20)
            {
                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    cumulativeDelta += (double)(bars[i].UpVolume - bars[i].DownVolume);
                }
                output.AddFeature("fg2_of_cumulative_delta", cumulativeDelta);
            }

            // Buy/Sell ratio
            output.AddFeature("fg2_of_buy_sell_ratio", SafeDiv((double)buyVolume, (double)sellVolume, 1.0));

            // Pressure ratio (aggressive buying vs selling)
            var pressureRatio = SafeDiv((double)buyVolume, (double)totalVolume);
            output.AddFeature("fg2_of_pressure_ratio", pressureRatio);

            // Trade intensity (trades per minute normalized)
            output.AddFeature("fg2_of_trade_intensity", (double)bar.TickVolume / 60.0);

            // Large order detection (using volume spikes)
            if (currentIndex >= 20)
            {
                var avgVolume = 0.0;
                for (int i = currentIndex - 19; i <= currentIndex - 1; i++)
                {
                    avgVolume += bars[i].TickVolume;
                }
                avgVolume /= 19;

                var largeOrderRatio = SafeDiv(bar.TickVolume, avgVolume, 1.0);
                output.AddFeature("fg2_of_large_order_ratio", largeOrderRatio);
            }

            // Aggressive ratio (market orders vs limit orders proxy)
            var aggressiveRatio = SafeDiv(
                Math.Abs((double)netFlow),
                (double)totalVolume
            );
            output.AddFeature("fg2_of_aggressive_ratio", aggressiveRatio);

            // VWAP calculation and deviation
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

                var vwap = SafeDiv(vwapNum, vwapDen);
                var vwapDev = SafeDiv((double)bar.Close - vwap, vwap) * 10000;
                output.AddFeature("fg2_of_vwap_deviation", vwapDev);

                // VWAP slope (momentum)
                if (currentIndex >= 25)
                {
                    double vwapNum2 = 0;
                    double vwapDen2 = 0;

                    for (int i = currentIndex - 24; i <= currentIndex - 5; i++)
                    {
                        var typical = (double)bars[i].TypicalPrice;
                        var vol = bars[i].TickVolume;
                        vwapNum2 += typical * vol;
                        vwapDen2 += vol;
                    }

                    var vwap2 = SafeDiv(vwapNum2, vwapDen2);
                    var vwapSlope = SafeDiv(vwap - vwap2, vwap2) * 10000;
                    output.AddFeature("fg2_of_vwap_slope", vwapSlope);
                }
            }

            // Microstructure metrics
            CalculateMicrostructureMetrics(output, bars, currentIndex);

            // Order book dynamics (simulated)
            CalculateOrderBookDynamics(output, bars, currentIndex);

            // Flow persistence metrics
            CalculateFlowPersistence(output, bars, currentIndex);

            // Update flow history
            _flowHistory.Add(new OrderFlowSnapshot
            {
                Timestamp = bar.Timestamp,
                TotalBuyVolume = buyVolume,
                TotalSellVolume = sellVolume,
                NetOrderFlow = netFlow,
                CumulativeDelta = (decimal)cumulativeDelta,
                BuyTradeCount = bar.UpTicks,
                SellTradeCount = bar.DownTicks,
                PressureRatio = (decimal)pressureRatio
            });
        }

        private void CalculateMicrostructureMetrics(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var bar = bars[currentIndex];

            // Effective spread (using average spread as proxy)
            output.AddFeature("fg2_of_effective_spread", (double)bar.AvgSpread * 10000);

            // Realized spread (price movement after trade)
            if (currentIndex >= 5)
            {
                var futurePrice = (double)bars[currentIndex - 5].Close;
                var currentMid = (double)((bar.High + bar.Low) / 2);
                var realizedSpread = Math.Abs(futurePrice - currentMid);
                output.AddFeature("fg2_of_realized_spread", realizedSpread * 10000);
            }

            // Price impact (volume-weighted price change)
            if (currentIndex >= 1)
            {
                var priceChange = Math.Abs((double)(bar.Close - bars[currentIndex - 1].Close));
                var avgVolume = (bar.TickVolume + bars[currentIndex - 1].TickVolume) / 2.0;
                var priceImpact = SafeDiv(priceChange * bar.TickVolume, avgVolume);
                output.AddFeature("fg2_of_price_impact", priceImpact * 10000);
            }

            // Quote imbalance
            var bidPressure = bar.Close - bar.Low;
            var askPressure = bar.High - bar.Close;
            var quoteImbalance = SafeDiv(
                (double)(bidPressure - askPressure),
                (double)(bidPressure + askPressure)
            );
            output.AddFeature("fg2_of_quote_imbalance", quoteImbalance);
        }

        private void CalculateOrderBookDynamics(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var bar = bars[currentIndex];

            // Simulated bid/ask depth changes based on volume distribution
            if (currentIndex >= 5)
            {
                var recentBuyVolume = 0.0;
                var recentSellVolume = 0.0;

                for (int i = currentIndex - 4; i <= currentIndex; i++)
                {
                    recentBuyVolume += (double)bars[i].UpVolume;
                    recentSellVolume += (double)bars[i].DownVolume;
                }

                // Depth changes (proxy)
                var bidDepthChange = SafeDiv(
                    recentBuyVolume - recentSellVolume,
                    recentBuyVolume + recentSellVolume
                );
                output.AddFeature("fg2_of_bid_depth_change", bidDepthChange);

                var askDepthChange = SafeDiv(
                    recentSellVolume - recentBuyVolume,
                    recentBuyVolume + recentSellVolume
                );
                output.AddFeature("fg2_of_ask_depth_change", askDepthChange);

                // Book imbalance
                var bookImbalance = SafeDiv(
                    recentBuyVolume - recentSellVolume,
                    recentBuyVolume + recentSellVolume
                );
                output.AddFeature("fg2_of_book_imbalance", bookImbalance);
            }

            // Micro price (volume-weighted mid price)
            var microPrice = (double)bar.Close + (double)bar.AvgSpread * SafeDiv((double)bar.UpVolume, (double)(bar.UpVolume + bar.DownVolume));
            output.AddFeature("fg2_of_micro_price", microPrice);
        }

        private void CalculateFlowPersistence(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (_flowHistory.Count < 10)
            {
                output.AddFeature("fg2_of_flow_autocorr", 0.0);
                output.AddFeature("fg2_of_flow_momentum", 0.0);
                output.AddFeature("fg2_of_flow_acceleration", 0.0);

                return;
            }

            // Flow autocorrelation
            var flows = _flowHistory.GetValues().Take(10).Select(f => (double)f.NetOrderFlow).ToList();
            var mean = flows.Average();

            double autocorr = 0;
            double variance = 0;

            for (int i = 0; i < flows.Count - 1; i++)
            {
                autocorr += (flows[i] - mean) * (flows[i + 1] - mean);
                variance += Math.Pow(flows[i] - mean, 2);
            }

            output.AddFeature("fg2_of_flow_autocorr", SafeDiv(autocorr, variance));

            // Flow momentum (recent vs older)
            if (_flowHistory.Count >= 20)
            {
                var recentFlow = _flowHistory.GetValues().Take(5).Average(f => (double)f.NetOrderFlow);
                var olderFlow = _flowHistory.GetValues().Skip(5).Take(5).Average(f => (double)f.NetOrderFlow);
                output.AddFeature("fg2_of_flow_momentum", recentFlow - olderFlow);
            }
            else
            {
                output.AddFeature("fg2_of_flow_momentum", 0.0);
            }

            // Flow acceleration
            if (_flowHistory.Count >= 3)
            {
                var flow1 = (double)_flowHistory[0].NetOrderFlow;
                var flow2 = (double)_flowHistory[1].NetOrderFlow;
                var flow3 = (double)_flowHistory[2].NetOrderFlow;

                var accel = (flow1 - flow2) - (flow2 - flow3);
                output.AddFeature("fg2_of_flow_acceleration", accel);
            }
            else
            {
                output.AddFeature("fg2_of_flow_acceleration", 0.0);
            }
        }

        public override void Reset()
        {
            _flowHistory.Clear();
            _bidLevels.Clear();
            _askLevels.Clear();
        }
    }
}
