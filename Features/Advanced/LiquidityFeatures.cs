using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Advanced
{

    public class LiquidityFeatures : BaseFeatureCalculator
    {
        public override string Name => "Liquidity";
        public override string Category => "Liquidity";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 11;

        private readonly RollingWindow<double> _spreadHistory = new(100);
        private readonly RollingWindow<double> _tickRateHistory = new(100);
        private readonly RollingWindow<double> _volumeHistory = new(100);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // ===== TICK RATE METRICS =====

            // Tick rate (ticks per minute)
            var tickRate = (double)bar.TickVolume / 1.0; // Assuming 1-minute bars
            output.AddFeature("fg2_liquidity_tick_rate", tickRate);
            _tickRateHistory.Add(tickRate);

            // Tick acceleration
            if (_tickRateHistory.Count >= 2)
            {
                var tickAcceleration = _tickRateHistory[0] - _tickRateHistory[1];
                output.AddFeature("fg2_liquidity_tick_acceleration", tickAcceleration);
            }
            else
            {
                output.AddFeature("fg2_liquidity_tick_acceleration", 0.0);
            }

            // Tick volatility
            if (_tickRateHistory.Count >= 20)
            {
                var tickVol = Math.Sqrt(_tickRateHistory.GetValues().Take(20)
                    .Select(t => Math.Pow(t - _tickRateHistory.GetValues().Take(20).Average(), 2))
                    .Average());
                output.AddFeature("fg2_liquidity_tick_volatility", tickVol);
            }
            else
            {
                output.AddFeature("fg2_liquidity_tick_volatility", 0.0);
            }

            // Tick clustering (concentration of ticks)
            if (currentIndex >= 10)
            {
                var recentTicks = 0;
                var totalTicks = 0;
                for (int i = currentIndex - 9; i <= currentIndex; i++)
                {
                    totalTicks += bars[i].TickVolume;
                    if (i >= currentIndex - 2)
                        recentTicks += bars[i].TickVolume;
                }
                var clustering = totalTicks > 0 ? (double)recentTicks / totalTicks : 0;
                output.AddFeature("fg2_liquidity_tick_clustering", clustering);
            }

            // ===== VOLUME PROFILE =====

            _volumeHistory.Add(bar.TickVolume);

            // Volume profile (distribution)
            if (currentIndex >= 20)
            {
                var volumes = new List<double>();
                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    volumes.Add(bars[i].TickVolume);
                }

                var volumeProfile = volumes.Max() - volumes.Min();
                output.AddFeature("fg2_liquidity_volume_profile", SafeDiv(volumeProfile, volumes.Average()));

                // Volume concentration (Herfindahl index)
                var totalVolume = volumes.Sum();
                var concentration = totalVolume > 0 ? volumes.Sum(v => Math.Pow(v / totalVolume, 2)) : 0;
                output.AddFeature("fg2_liquidity_volume_concentration", concentration);

                // Volume dispersion
                var volumeDispersion = Math.Sqrt(volumes.Select(v => Math.Pow(v - volumes.Average(), 2)).Average());
                output.AddFeature("fg2_liquidity_volume_dispersion", volumeDispersion);
            }

            // Relative volume
            if (_volumeHistory.Count >= 50)
            {
                var avgVolume = _volumeHistory.GetValues().Take(50).Average();
                var relativeVolume = SafeDiv(bar.TickVolume, avgVolume);
                output.AddFeature("fg2_liquidity_relative_volume", relativeVolume);
            }
            else
            {
                output.AddFeature("fg2_liquidity_relative_volume", 1.0);
            }

            // ===== PRICE DISPERSION & EFFICIENCY =====

            // Price dispersion
            if (currentIndex >= 10)
            {
                var prices = new List<double>();
                for (int i = currentIndex - 9; i <= currentIndex; i++)
                {
                    prices.Add((double)bars[i].Close);
                }
                var priceDisp = prices.Max() - prices.Min();
                output.AddFeature("fg2_liquidity_price_dispersion", SafeDiv(priceDisp, prices.Average()) * 10000);
            }

            // Price efficiency (how directly price moves)
            if (currentIndex >= 10)
            {
                var netMove = Math.Abs((double)(bar.Close - bars[currentIndex - 9].Close));
                var totalMove = 0.0;
                for (int i = currentIndex - 8; i <= currentIndex; i++)
                {
                    totalMove += Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
                }
                var efficiency = SafeDiv(netMove, totalMove);
                output.AddFeature("fg2_liquidity_price_efficiency", efficiency);
            }

            // Effective tick (price move per tick)
            var effectiveTick = bar.TickVolume > 0 ?
                (double)(bar.High - bar.Low) / bar.TickVolume * 10000 : 0;
            output.AddFeature("fg2_liquidity_effective_tick", effectiveTick);

            // Price impact (simplified Kyle's lambda)
            if (currentIndex >= 1)
            {
                var priceChange = Math.Abs((double)(bar.Close - bars[currentIndex - 1].Close));
                var priceImpact = SafeDiv(priceChange * 10000, bar.TickVolume);
                output.AddFeature("fg2_liquidity_price_impact", priceImpact);
            }

            // ===== MARKET DEPTH PROXIES =====

            // Depth proxy (based on spread and volume)
            var depthProxy = bar.TickVolume / Math.Max(0.0001, (double)bar.AvgSpread * 10000);
            output.AddFeature("fg2_liquidity_depth_proxy", Math.Log(1 + depthProxy));

            // Resilience (how quickly price recovers)
            if (currentIndex >= 5)
            {
                var midPrice = (double)((bar.High + bar.Low) / 2);
                var prevMidPrice = (double)((bars[currentIndex - 5].High + bars[currentIndex - 5].Low) / 2);
                var priceDeviation = Math.Abs(midPrice - prevMidPrice);

                // Check recovery
                var recovered = 0.0;
                for (int i = currentIndex - 4; i <= currentIndex; i++)
                {
                    var mp = (double)((bars[i].High + bars[i].Low) / 2);
                    if (Math.Abs(mp - prevMidPrice) < priceDeviation * 0.5)
                    {
                        recovered = 1.0;
                        break;
                    }
                }
                output.AddFeature("fg2_liquidity_resilience", recovered);
            }

            // Tightness (relative spread)
            var tightness = SafeDiv((double)bar.AvgSpread, close) * 10000;
            output.AddFeature("fg2_liquidity_tightness", tightness);

            // ===== ADVANCED LIQUIDITY MEASURES =====

            // Amihud Illiquidity Measure
            if (currentIndex >= 20)
            {
                double amihudSum = 0;
                int validDays = 0;

                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    var ret = i > 0 ? Math.Abs((double)(bars[i].Close - bars[i - 1].Close) / (double)bars[i - 1].Close) : 0;
                    var dollarVolume = bars[i].TickVolume * (double)bars[i].Close;

                    if (dollarVolume > 0)
                    {
                        amihudSum += ret / dollarVolume * 1000000; // Scale factor
                        validDays++;
                    }
                }

                var amihud = validDays > 0 ? amihudSum / validDays : 0;
                output.AddFeature("fg2_liquidity_amihud_illiquidity", amihud);
            }

            // Roll Measure (effective spread estimator)
            if (currentIndex >= 2)
            {
                var priceChange = (double)(bar.Close - bars[currentIndex - 1].Close);
                var prevPriceChange = (double)(bars[currentIndex - 1].Close - bars[currentIndex - 2].Close);
                var rollMeasure = 2 * Math.Sqrt(Math.Max(0, -priceChange * prevPriceChange));
                output.AddFeature("fg2_liquidity_roll_measure", rollMeasure * 10000);
            }

            // Kyle's Lambda (simplified price impact)
            if (currentIndex >= 10)
            {
                var priceChanges = new List<double>();
                var volumes = new List<double>();

                for (int i = currentIndex - 9; i <= currentIndex; i++)
                {
                    priceChanges.Add((double)(bars[i].Close - bars[i - 1].Close));
                    volumes.Add((double)bars[i].UpVolume - (double)bars[i].DownVolume); // Net volume
                }

                // Simple regression of price change on net volume
                var lambda = CalculateKyleLambda(priceChanges, volumes);
                output.AddFeature("fg2_liquidity_kyle_lambda", lambda * 10000);
            }

            // Hasbrouck Measure (information share)
            if (currentIndex >= 30)
            {
                var hasbrouck = CalculateHasbrouckMeasure(bars, currentIndex);
                output.AddFeature("fg2_liquidity_hasbrouck_measure", hasbrouck);
            }
        }

        private double CalculateKyleLambda(List<double> priceChanges, List<double> volumes)
        {
            var n = priceChanges.Count;
            if (n < 2) return 0;

            var avgPrice = priceChanges.Average();
            var avgVolume = volumes.Average();

            double numerator = 0;
            double denominator = 0;

            for (int i = 0; i < n; i++)
            {
                numerator += (volumes[i] - avgVolume) * (priceChanges[i] - avgPrice);
                denominator += Math.Pow(volumes[i] - avgVolume, 2);
            }

            return denominator > 0 ? numerator / denominator : 0;
        }

        private double CalculateHasbrouckMeasure(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified Hasbrouck information share measure
            // Based on variance decomposition of price changes

            var shortTermVar = 0.0;
            var longTermVar = 0.0;

            // Short-term variance (1-minute returns)
            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                var ret = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                shortTermVar += ret * ret;
            }
            shortTermVar /= 10;

            // Long-term variance (5-minute returns)
            for (int i = currentIndex - 29; i <= currentIndex; i += 5)
            {
                if (i >= 5)
                {
                    var ret = Math.Log((double)bars[i].Close / (double)bars[i - 5].Close);
                    longTermVar += ret * ret;
                }
            }
            longTermVar /= 6;

            // Information share (ratio of variances)
            return SafeDiv(shortTermVar, longTermVar);
        }

        public override void Reset()
        {
            _spreadHistory.Clear();
            _tickRateHistory.Clear();
            _volumeHistory.Clear();
        }
    }
}
