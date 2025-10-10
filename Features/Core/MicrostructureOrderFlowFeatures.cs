using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Market microstructure and order flow features optimized for directional prediction
    /// Focuses on volume imbalances, order flow patterns, and liquidity dynamics
    /// </summary>
    public class MicrostructureOrderFlowFeatures : BaseFeatureCalculator
    {
        public override string Name => "MicrostructureOrderFlow";
        public override string Category => "Microstructure";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 3;

        private readonly RollingWindow<OrderFlowSnapshot> _flowHistory = new(100);
        private readonly RollingWindow<double> _spreadHistory = new(50);
        private readonly RollingWindow<double> _imbalanceHistory = new(50);

        private class OrderFlowSnapshot
        {
            public double NetFlow { get; set; }
            public double BuyPressure { get; set; }
            public double SellPressure { get; set; }
            public double ImbalanceRatio { get; set; }
            public double VolumeRate { get; set; }
        }

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // ===== 1. ORDER FLOW IMBALANCE FEATURES =====
            // Critical for predicting short-term direction

            // Net order flow (buy - sell volume)
            var netFlow = (double)(bar.UpVolume - bar.DownVolume);
            var totalVolume = (double)(bar.UpVolume + bar.DownVolume);

            // Normalized order flow imbalance [-1, 1]
            var flowImbalance = totalVolume > 0 ? netFlow / totalVolume : 0;
            output.AddFeature("03_micro_flow_imbalance", flowImbalance);

            // Order flow imbalance signal
            var imbalanceSignal = CreateDirectionalSignal(flowImbalance, 0.3, -0.3);
            output.AddFeature("03_micro_imbalance_signal", imbalanceSignal);

            // Cumulative Volume Delta (CVD)
            var cvd = CalculateCVD(bars, currentIndex, 10);
            var cvdNormalized = NormalizeCVD(cvd, bars, currentIndex);
            output.AddFeature("03_micro_cvd_normalized", cvdNormalized);
            output.AddFeature("03_micro_cvd_signal", CreateDirectionalSignal(cvdNormalized, 0.5, -0.5));

            // Volume-weighted order flow
            var vwof = CalculateVolumeWeightedOrderFlow(bars, currentIndex);
            output.AddFeature("03_micro_vwof", vwof);

            // Order flow acceleration
            var flowAcceleration = CalculateFlowAcceleration(netFlow, _flowHistory);
            output.AddFeature("03_micro_flow_acceleration", flowAcceleration);

            _imbalanceHistory.Add(flowImbalance);

            // ===== 2. VOLUME PRESSURE FEATURES =====
            // Indicates buying/selling pressure intensity

            // Buy/sell pressure ratio
            var buyPressure = bar.UpVolume > 0 ? (double)bar.UpVolume / Math.Max(1, bar.TickVolume) : 0;
            var sellPressure = bar.DownVolume > 0 ? (double)bar.DownVolume / Math.Max(1, bar.TickVolume) : 0;

            output.AddFeature("03_micro_buy_pressure", buyPressure);
            output.AddFeature("03_micro_sell_pressure", sellPressure);

            // Pressure differential (key directional indicator)
            var pressureDiff = buyPressure - sellPressure;
            output.AddFeature("03_micro_pressure_diff", pressureDiff);
            output.AddFeature("03_micro_pressure_signal", CreateDirectionalSignal(pressureDiff, 0.2, -0.2));

            // Large order detection (volume spike)
            var volumeSpike = DetectVolumeSpike(bar, bars, currentIndex);
            output.AddFeature("03_micro_volume_spike", volumeSpike);
            output.AddFeature("03_micro_spike_direction", volumeSpike * Math.Sign(netFlow));

            // ===== 3. SPREAD & LIQUIDITY FEATURES =====
            // Spread patterns indicate market stress and direction

            var spreadBps = (double)bar.AvgSpread * 10000 / close;  // Spread in basis points
            _spreadHistory.Add(spreadBps);

            // Normalized spread (z-score)
            var spreadZScore = CalculateSpreadZScore(spreadBps, _spreadHistory);
            output.AddFeature("03_micro_spread_zscore", spreadZScore);

            // Spread regime (tight/normal/wide)
            var spreadRegime = spreadZScore > 1.5 ? 1.0 :    // Wide spread (uncertainty)
                              spreadZScore < -1.5 ? -1.0 :   // Tight spread (confidence)
                              0.0;
            output.AddFeature("03_micro_spread_regime", spreadRegime);

            // Spread-volume relationship (liquidity indicator)
            var spreadVolumeRatio = SafeDiv(spreadBps, Math.Log(1 + bar.TickVolume));
            output.AddFeature("03_micro_spread_volume_ratio", Sigmoid(spreadVolumeRatio - 1));

            // Effective spread (price impact proxy)
            var effectiveSpread = CalculateEffectiveSpread(bar, bars, currentIndex);
            output.AddFeature("03_micro_effective_spread", effectiveSpread);

            // ===== 4. TICK DYNAMICS FEATURES =====
            // Tick patterns reveal order flow dynamics

            // Tick intensity (activity level)
            var tickRate = (double)bar.TickVolume / 60.0;  // Ticks per second (assuming 1-min bar)
            var tickIntensity = CalculateTickIntensity(tickRate, bars, currentIndex);
            output.AddFeature("03_micro_tick_intensity", tickIntensity);

            // Tick direction ratio
            var tickDirectionRatio = bar.TickVolume > 0 ? (double)(bar.UpVolume - bar.DownVolume) / bar.TickVolume : 0;
            output.AddFeature("03_micro_tick_direction", tickDirectionRatio);

            // Tick clustering (concentration of activity)
            var tickClustering = CalculateTickClustering(bars, currentIndex);
            output.AddFeature("03_micro_tick_clustering", tickClustering);

            // ===== 5. VWAP & PRICE EFFICIENCY =====
            // VWAP deviation indicates institutional activity

            var vwap = CalculateVWAP(bars, currentIndex, 20);
            var vwapDeviation = SafeDiv(close - vwap, vwap) * 10000;  // In basis points
            var vwapSignal = CreateDirectionalSignal(vwapDeviation, 10, -10);  // ±10 bps threshold

            output.AddFeature("03_micro_vwap_deviation", Sigmoid(vwapDeviation / 20));
            output.AddFeature("03_micro_vwap_signal", vwapSignal);

            // VWAP pull strength (mean reversion to VWAP)
            var vwapPull = CalculateVWAPPull(close, vwap, bars, currentIndex);
            output.AddFeature("03_micro_vwap_pull", vwapPull);

            // Price efficiency (how directly price moves)
            var priceEfficiency = CalculatePriceEfficiency(bars, currentIndex);
            output.AddFeature("03_micro_price_efficiency", priceEfficiency);

            // ===== 6. MARKET DEPTH PROXIES =====
            // Inferred market depth from price/volume dynamics

            // Depth imbalance proxy
            var depthImbalance = CalculateDepthImbalance(bar, bars, currentIndex);
            output.AddFeature("03_micro_depth_imbalance", depthImbalance);

            // Kyle's lambda (price impact)
            var kyleLambda = CalculateKyleLambda(bars, currentIndex);
            output.AddFeature("03_micro_kyle_lambda", Sigmoid(kyleLambda * 1000));

            // Amihud illiquidity
            var amihud = CalculateAmihudIlliquidity(bars, currentIndex);
            output.AddFeature("03_micro_amihud_illiquidity", Sigmoid(amihud * 100));

            // ===== 7. MICROSTRUCTURE PATTERNS =====
            // Specific patterns that precede directional moves

            // Absorption pattern (large volume, small price change)
            var absorption = DetectAbsorptionPattern(bar, bars, currentIndex);
            output.AddFeature("03_micro_absorption", absorption);

            // Iceberg order detection
            var iceberg = DetectIcebergPattern(bars, currentIndex);
            output.AddFeature("03_micro_iceberg_pattern", iceberg);

            // Stop hunt pattern
            var stopHunt = DetectStopHuntPattern(bars, currentIndex);
            output.AddFeature("03_micro_stop_hunt", stopHunt);

            // ===== 8. COMPOSITE MICROSTRUCTURE SIGNALS =====

            // Order flow composite
            var flowComposite = CreateCompositeSignal(
                (flowImbalance, 0.25),
                (cvdNormalized, 0.25),
                (pressureDiff, 0.25),
                (vwapSignal, 0.25)
            );
            output.AddFeature("03_micro_flow_composite", flowComposite);

            // Liquidity composite
            var liquidityComposite = CreateCompositeSignal(
                (-spreadZScore / 3, 0.3),  // Inverted: tight spread = good liquidity
                (tickIntensity, 0.3),
                (priceEfficiency, 0.2),
                (-amihud, 0.2)  // Inverted: low illiquidity = good
            );
            output.AddFeature("03_micro_liquidity_composite", liquidityComposite);

            // Master microstructure signal
            var microMaster = CreateCompositeSignal(
                (flowComposite, 0.4),
                (liquidityComposite * 0.5, 0.2),  // Liquidity supports direction
                (depthImbalance, 0.2),
                (absorption, 0.2)
            );
            output.AddFeature("03_micro_master_signal", microMaster);

            // Signal quality (based on microstructure health)
            var signalQuality = CalculateMicrostructureQuality(spreadZScore, tickIntensity, priceEfficiency, Math.Abs(flowImbalance));
            output.AddFeature("03_micro_signal_quality", signalQuality);

            // Update history
            _flowHistory.Add(new OrderFlowSnapshot
            {
                NetFlow = netFlow,
                BuyPressure = buyPressure,
                SellPressure = sellPressure,
                ImbalanceRatio = flowImbalance,
                VolumeRate = tickRate
            });
        }

        // ===== CALCULATION METHODS =====

        private double CalculateCVD(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            double cvd = 0;
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                cvd += (double)(bars[i].UpVolume - bars[i].DownVolume);
            }
            return cvd;
        }

        private double NormalizeCVD(double cvd, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Normalize by total volume
            double totalVolume = 0;
            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                totalVolume += bars[i].TickVolume;
            }

            return totalVolume > 0 ? Sigmoid(cvd / totalVolume) : 0;
        }

        private double CalculateVolumeWeightedOrderFlow(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            double vwof = 0;
            double totalVolume = 0;

            for (int i = currentIndex - 4; i <= currentIndex; i++)
            {
                var flow = (double)(bars[i].UpVolume - bars[i].DownVolume);
                var volume = bars[i].TickVolume;
                vwof += flow * volume;
                totalVolume += volume;
            }

            return totalVolume > 0 ? vwof / totalVolume / totalVolume : 0;
        }

        private double CalculateFlowAcceleration(double currentFlow, RollingWindow<OrderFlowSnapshot> history)
        {
            if (history.Count < 3) return 0;

            var flow1 = currentFlow;
            var flow2 = history[0].NetFlow;
            var flow3 = history[1].NetFlow;

            var accel = (flow1 - flow2) - (flow2 - flow3);
            return Sigmoid(accel / 100);
        }

        private double DetectVolumeSpike(OhlcBar bar, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return 0;

            // Calculate average volume
            double avgVolume = 0;
            for (int i = currentIndex - 19; i <= currentIndex - 1; i++)
            {
                avgVolume += bars[i].TickVolume;
            }
            avgVolume /= 19;

            var spikeRatio = SafeDiv(bar.TickVolume, avgVolume);
            return spikeRatio > 2.0 ? 1.0 :   // Large spike
                   spikeRatio > 1.5 ? 0.5 :   // Moderate spike
                   0.0;
        }

        private double CalculateSpreadZScore(double currentSpread, RollingWindow<double> history)
        {
            if (history.Count < 20) return 0;

            var values = history.GetValues().Take(20).ToList();
            var mean = values.Average();
            var stdDev = Math.Sqrt(values.Select(v => Math.Pow(v - mean, 2)).Average());

            return CalculateZScore(currentSpread, mean, stdDev);
        }

        private double CalculateEffectiveSpread(OhlcBar bar, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Effective spread = 2 * |execution price - mid price|
            // Using close as execution price proxy
            var midPrice = (double)((bar.High + bar.Low) / 2);
            var effectiveSpread = 2 * Math.Abs((double)bar.Close - midPrice);

            // Normalize by price level
            return SafeDiv(effectiveSpread, midPrice) * 10000;  // In basis points
        }

        private double CalculateTickIntensity(double currentRate, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var rates = new List<double>();
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                rates.Add((double)bars[i].TickVolume / 60.0);
            }

            var percentile = CalculatePercentileRank(currentRate, rates);
            return (percentile - 50) / 50;  // Normalize to [-1, 1]
        }

        private double CalculateTickClustering(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 5) return 0;

            // Measure concentration of ticks in recent vs older bars
            double recentTicks = 0;
            double totalTicks = 0;

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                totalTicks += bars[i].TickVolume;
                if (i >= currentIndex - 2)
                    recentTicks += bars[i].TickVolume;
            }

            return totalTicks > 0 ? (recentTicks / totalTicks - 0.3) / 0.7 : 0;  // Normalize
        }

        private double CalculateVWAP(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            double priceVolume = 0;
            double totalVolume = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var typicalPrice = (double)bars[i].TypicalPrice;
                var volume = bars[i].TickVolume;
                priceVolume += typicalPrice * volume;
                totalVolume += volume;
            }

            return totalVolume > 0 ? priceVolume / totalVolume : (double)bars[currentIndex].Close;
        }

        private double CalculateVWAPPull(double price, double vwap, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var deviation = price - vwap;
            var atr = CalculateATR(bars, currentIndex);

            // Normalize deviation by ATR
            var normalizedDev = SafeDiv(Math.Abs(deviation), atr);

            // Strong pull if far from VWAP
            if (normalizedDev > 2)
                return -Math.Sign(deviation);  // Expect reversion

            return 0;
        }

        private double CalculatePriceEfficiency(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 10) return 0.5;

            var netMove = Math.Abs((double)(bars[currentIndex].Close - bars[currentIndex - 9].Close));
            double totalMove = 0;

            for (int i = currentIndex - 8; i <= currentIndex; i++)
            {
                totalMove += Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
            }

            return SafeDiv(netMove, totalMove);
        }

        private double CalculateDepthImbalance(OhlcBar bar, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Infer depth imbalance from price response to volume
            var priceChange = currentIndex > 0 ?
                (double)(bar.Close - bars[currentIndex - 1].Close) : 0;
            var volume = bar.TickVolume;

            // Large volume with small price change = balanced depth
            // Small volume with large price change = imbalanced depth
            var priceResponse = SafeDiv(Math.Abs(priceChange) * 10000, Math.Log(1 + volume));

            return Sigmoid((priceResponse - 5) / 5) * Math.Sign(priceChange);
        }

        private double CalculateKyleLambda(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 10) return 0;

            var priceChanges = new List<double>();
            var netVolumes = new List<double>();

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                priceChanges.Add((double)(bars[i].Close - bars[i - 1].Close));
                netVolumes.Add((double)(bars[i].UpVolume - bars[i].DownVolume));
            }

            // Simple regression of price on net volume
            var avgPrice = priceChanges.Average();
            var avgVolume = netVolumes.Average();

            double numerator = 0;
            double denominator = 0;

            for (int i = 0; i < priceChanges.Count; i++)
            {
                numerator += (netVolumes[i] - avgVolume) * (priceChanges[i] - avgPrice);
                denominator += Math.Pow(netVolumes[i] - avgVolume, 2);
            }

            return SafeDiv(numerator, denominator);
        }

        private double CalculateAmihudIlliquidity(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 10) return 0;

            double sum = 0;
            int count = 0;

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                var returns = Math.Abs((double)(bars[i].Close - bars[i - 1].Close) / (double)bars[i - 1].Close);
                var dollarVolume = bars[i].TickVolume * (double)bars[i].Close;

                if (dollarVolume > 0)
                {
                    sum += returns / dollarVolume * 1000000;
                    count++;
                }
            }

            return count > 0 ? sum / count : 0;
        }

        private double DetectAbsorptionPattern(OhlcBar bar, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 5) return 0;

            // High volume with small price change = absorption
            var avgVolume = 0.0;
            var avgRange = 0.0;

            for (int i = currentIndex - 4; i <= currentIndex - 1; i++)
            {
                avgVolume += bars[i].TickVolume;
                avgRange += (double)(bars[i].High - bars[i].Low);
            }
            avgVolume /= 4;
            avgRange /= 4;

            var volumeRatio = SafeDiv(bar.TickVolume, avgVolume);
            var rangeRatio = SafeDiv((double)(bar.High - bar.Low), avgRange);

            // Absorption if high volume but small range
            if (volumeRatio > 1.5 && rangeRatio < 0.7)
            {
                // Direction based on close position in range
                var closePosition = SafeDiv(
                    (double)(bar.Close - bar.Low),
                    (double)(bar.High - bar.Low));

                return closePosition > 0.7 ? 1.0 :   // Bullish absorption
                       closePosition < 0.3 ? -1.0 :  // Bearish absorption
                       0.0;
            }

            return 0;
        }

        private double DetectIcebergPattern(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 10) return 0;

            // Consistent volume at same price level = potential iceberg
            var currentPrice = (double)bars[currentIndex].Close;
            int similarPriceCount = 0;
            double volumeAtLevel = 0;

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                if (Math.Abs((double)bars[i].Close - currentPrice) < currentPrice * 0.0001)  // Within 1 pip
                {
                    similarPriceCount++;
                    volumeAtLevel += bars[i].TickVolume;
                }
            }

            if (similarPriceCount >= 3 && volumeAtLevel > bars[currentIndex].TickVolume * 5)
            {
                // Iceberg detected, direction based on recent break
                if (currentIndex > 0)
                {
                    var priceChange = (double)(bars[currentIndex].Close - bars[currentIndex - 1].Close);
                    return Math.Sign(priceChange) * 0.5;
                }
            }

            return 0;
        }

        private double DetectStopHuntPattern(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 3) return 0;

            var bar = bars[currentIndex];
            var prevBar = bars[currentIndex - 1];

            // Stop hunt: spike beyond previous high/low then reversal
            var spikeUp = bar.High > prevBar.High && bar.Close < prevBar.High;
            var spikeDown = bar.Low < prevBar.Low && bar.Close > prevBar.Low;

            if (spikeUp)
                return -0.5;  // Bearish after stop hunt above
            if (spikeDown)
                return 0.5;   // Bullish after stop hunt below

            return 0;
        }

        private double CalculateATR(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 14) return (double)(bars[currentIndex].High - bars[currentIndex].Low);

            double sum = 0;
            for (int i = currentIndex - 13; i <= currentIndex; i++)
            {
                var tr = Math.Max((double)(bars[i].High - bars[i].Low),
                        Math.Max(Math.Abs((double)(bars[i].High - bars[i - 1].Close)),
                                Math.Abs((double)(bars[i].Low - bars[i - 1].Close))));
                sum += tr;
            }
            return sum / 14;
        }

        private double CalculateMicrostructureQuality(double spreadZ, double tickIntensity,
            double efficiency, double flowStrength)
        {
            // Quality is high when:
            // - Spread is normal (|z| < 1)
            // - Tick intensity is high
            // - Price efficiency is high
            // - Flow imbalance is strong

            var spreadQuality = Math.Max(0, 1 - Math.Abs(spreadZ) / 3);
            var intensityQuality = (tickIntensity + 1) / 2;  // Convert from [-1,1] to [0,1]
            var efficiencyQuality = efficiency;
            var flowQuality = flowStrength;

            return (spreadQuality * 0.2 + intensityQuality * 0.2 +
                   efficiencyQuality * 0.3 + flowQuality * 0.3);
        }

        public override void Reset()
        {
            _flowHistory.Clear();
            _spreadHistory.Clear();
            _imbalanceHistory.Clear();
        }
    }
}