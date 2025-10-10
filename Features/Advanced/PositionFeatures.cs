using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Core;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Advanced
{
    /// <summary>
    /// Features specifically designed for LONG/SHORT position analysis,
    /// trade setup quality, entry/exit signals, and position management
    /// </summary>
    public class PositionFeatures : BaseFeatureCalculator
    {
        public override string Name => "Position";
        public override string Category => "Position_Analysis";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 5;

        private readonly RollingWindow<PositionSnapshot> _positionHistory = new(100);
        private readonly RollingWindow<TradeSetup> _setupHistory = new(50);

        // Scalping parameters (from label config: 3.5/2.5 pips)
        private const double TRAILING_STOP_ACTIVATION_PIPS = 3.5;
        private const double TRAILING_STOP_DISTANCE_PIPS = 2.5;
        private const double MAX_DRAWDOWN_PIPS = 10.0;

        public class PositionSnapshot
        {
            public DateTime Timestamp { get; set; }
            public int RecommendedPosition { get; set; } // 1=LONG, -1=SHORT, 0=NEUTRAL
            public double Confidence { get; set; }
            public double LongQuality { get; set; }
            public double ShortQuality { get; set; }
            public double RiskReward { get; set; }
        }

        public class TradeSetup
        {
            public DateTime Timestamp { get; set; }
            public int Direction { get; set; }
            public double EntryQuality { get; set; }
            public double ExpectedProfit { get; set; }
            public double ExpectedRisk { get; set; }
        }

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // ===== LONG POSITION QUALITY =====
            var longQuality = CalculateLongPositionQuality(bars, currentIndex);
            output.AddFeature("pos_long_quality", longQuality.quality);
            output.AddFeature("pos_long_entry_score", longQuality.entryScore);
            output.AddFeature("pos_long_risk_reward", longQuality.riskReward);
            output.AddFeature("pos_long_success_prob", longQuality.successProbability);

            // ===== SHORT POSITION QUALITY =====
            var shortQuality = CalculateShortPositionQuality(bars, currentIndex);
            output.AddFeature("pos_short_quality", shortQuality.quality);
            output.AddFeature("pos_short_entry_score", shortQuality.entryScore);
            output.AddFeature("pos_short_risk_reward", shortQuality.riskReward);
            output.AddFeature("pos_short_success_prob", shortQuality.successProbability);

            // ===== POSITION RECOMMENDATION =====
            var (recommendedPos, confidence) = DeterminePositionRecommendation(longQuality, shortQuality);
            output.AddFeature("pos_recommendation", recommendedPos);
            output.AddFeature("pos_recommendation_confidence", confidence);

            // ===== ENTRY SIGNALS =====
            var longEntry = DetectLongEntrySignal(bars, currentIndex);
            output.AddFeature("pos_long_entry_signal", longEntry.signal);
            output.AddFeature("pos_long_entry_strength", longEntry.strength);
            output.AddFeature("pos_long_entry_confirmation", longEntry.confirmation);

            var shortEntry = DetectShortEntrySignal(bars, currentIndex);
            output.AddFeature("pos_short_entry_signal", shortEntry.signal);
            output.AddFeature("pos_short_entry_strength", shortEntry.strength);
            output.AddFeature("pos_short_entry_confirmation", shortEntry.confirmation);

            // ===== EXIT SIGNALS =====
            output.AddFeature("pos_long_exit_signal", DetectLongExitSignal(bars, currentIndex));
            output.AddFeature("pos_short_exit_signal", DetectShortExitSignal(bars, currentIndex));

            // ===== TRAILING STOP ANALYSIS =====
            var longTrailing = AnalyzeTrailingStopLong(bars, currentIndex);
            output.AddFeature("pos_long_trailing_active", longTrailing.wouldActivate ? 1.0 : 0.0);
            output.AddFeature("pos_long_profit_potential", longTrailing.potentialProfit);
            output.AddFeature("pos_long_max_favorable", longTrailing.maxFavorable);

            var shortTrailing = AnalyzeTrailingStopShort(bars, currentIndex);
            output.AddFeature("pos_short_trailing_active", shortTrailing.wouldActivate ? 1.0 : 0.0);
            output.AddFeature("pos_short_profit_potential", shortTrailing.potentialProfit);
            output.AddFeature("pos_short_max_favorable", shortTrailing.maxFavorable);

            // ===== RISK ANALYSIS =====
            var riskMetrics = CalculateRiskMetrics(bars, currentIndex);
            output.AddFeature("pos_downside_risk", riskMetrics.downsideRisk);
            output.AddFeature("pos_upside_potential", riskMetrics.upsidePotential);
            output.AddFeature("pos_risk_asymmetry", riskMetrics.asymmetry);
            output.AddFeature("pos_stop_distance", riskMetrics.stopDistance);

            // ===== MARKET STRUCTURE FOR POSITIONS =====
            var structure = AnalyzeMarketStructure(bars, currentIndex);
            output.AddFeature("pos_support_strength", structure.supportStrength);
            output.AddFeature("pos_resistance_strength", structure.resistanceStrength);
            output.AddFeature("pos_trend_alignment", structure.trendAlignment);
            output.AddFeature("pos_momentum_alignment", structure.momentumAlignment);

            // ===== OPTIMAL ENTRY LEVELS =====
            var optimalEntry = FindOptimalEntryLevels(bars, currentIndex);
            output.AddFeature("pos_optimal_long_entry", optimalEntry.longEntry);
            output.AddFeature("pos_optimal_short_entry", optimalEntry.shortEntry);
            output.AddFeature("pos_distance_to_long_entry", SafeDiv(optimalEntry.longEntry - close, close) * 10000);
            output.AddFeature("pos_distance_to_short_entry", SafeDiv(close - optimalEntry.shortEntry, close) * 10000);

            // ===== POSITION HOLDING DURATION ESTIMATE =====
            output.AddFeature("pos_expected_long_duration", EstimateHoldingDuration(bars, currentIndex, true));
            output.AddFeature("pos_expected_short_duration", EstimateHoldingDuration(bars, currentIndex, false));

            // ===== MULTI-TIMEFRAME POSITION ALIGNMENT =====
            if (currentIndex >= 50)
            {
                var alignment = CalculateMultiTimeframeAlignment(bars, currentIndex);
                output.AddFeature("pos_mtf_long_alignment", alignment.longAlignment);
                output.AddFeature("pos_mtf_short_alignment", alignment.shortAlignment);
                output.AddFeature("pos_mtf_consensus", alignment.consensus);
            }

            // ===== POSITION SIZING RECOMMENDATION =====
            var sizing = RecommendPositionSize(bars, currentIndex, longQuality, shortQuality);
            output.AddFeature("pos_recommended_size_long", sizing.longSize);
            output.AddFeature("pos_recommended_size_short", sizing.shortSize);
            output.AddFeature("pos_size_confidence", sizing.confidence);

            // ===== TRADE EXPECTANCY =====
            output.AddFeature("pos_long_expectancy", CalculateTradeExpectancy(bars, currentIndex, true));
            output.AddFeature("pos_short_expectancy", CalculateTradeExpectancy(bars, currentIndex, false));

            // ===== POSITION CORRELATION =====
            // How correlated is the position with recent successful setups
            if (_setupHistory.Count >= 10)
            {
                var correlation = CalculateSetupCorrelation(bars, currentIndex);
                output.AddFeature("pos_setup_correlation", correlation);
            }

            // ===== ADVERSE SELECTION RISK =====
            output.AddFeature("pos_adverse_selection", CalculateAdverseSelectionRisk(bars, currentIndex));

            // ===== SLIPPAGE ESTIMATE =====
            var slippage = EstimateSlippage(bars, currentIndex);
            output.AddFeature("pos_expected_slippage", slippage);

            // ===== WIN PROBABILITY =====
            output.AddFeature("pos_long_win_probability", EstimateWinProbability(bars, currentIndex, true));
            output.AddFeature("pos_short_win_probability", EstimateWinProbability(bars, currentIndex, false));

            // Update histories
            _positionHistory.Add(new PositionSnapshot
            {
                Timestamp = bar.Timestamp,
                RecommendedPosition = (int)recommendedPos,
                Confidence = confidence,
                LongQuality = longQuality.quality,
                ShortQuality = shortQuality.quality,
                RiskReward = (longQuality.riskReward + shortQuality.riskReward) / 2
            });
        }

        private (double quality, double entryScore, double riskReward, double successProbability) CalculateLongPositionQuality(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var scores = new List<double>();

            // 1. Trend alignment
            if (currentIndex >= 50)
            {
                var ema9 = CalculateEMA(bars, 9, currentIndex);
                var ema21 = CalculateEMA(bars, 21, currentIndex);
                var ema50 = CalculateEMA(bars, 50, currentIndex);

                if (ema9 > ema21 && ema21 > ema50) scores.Add(1.0);
                else if (ema9 > ema21) scores.Add(0.6);
                else scores.Add(0.2);
            }

            // 2. Momentum
            var rsi = CalculateRSI(bars, 14, currentIndex);
            if (rsi > 30 && rsi < 70) scores.Add(1.0);
            else if (rsi >= 70) scores.Add(0.5);
            else scores.Add(0.3);

            // 3. Volume confirmation
            var currentVol = (double)bars[currentIndex].TickVolume;
            var avgVol = 0.0;
            for (int i = currentIndex - 19; i <= currentIndex - 1; i++)
            {
                avgVol += bars[i].TickVolume;
            }
            avgVol /= 19;

            if (currentVol > avgVol * 1.2) scores.Add(1.0);
            else if (currentVol > avgVol) scores.Add(0.7);
            else scores.Add(0.4);

            // 4. Support nearby
            var supportDist = FindNearestSupport(bars, currentIndex);
            if (supportDist < 5.0) scores.Add(1.0); // Within 5 pips
            else if (supportDist < 10.0) scores.Add(0.7);
            else scores.Add(0.3);

            // 5. Volatility favorable
            var atr = CalculateATR(bars, 14, currentIndex);
            var avgAtr = 0.0;
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                avgAtr += CalculateATR(bars, 14, i);
            }
            avgAtr /= 20;

            if (atr < avgAtr * 1.5) scores.Add(1.0); // Not too volatile
            else scores.Add(0.5);

            var quality = scores.Average();
            var entryScore = quality * (1.0 + (bars[currentIndex].UpVolume / (double)(bars[currentIndex].UpVolume + bars[currentIndex].DownVolume)));
            var riskReward = CalculateLongRiskReward(bars, currentIndex);
            var successProb = quality * 0.8; // Dampened

            return (quality, entryScore / 2, riskReward, successProb);
        }

        private (double quality, double entryScore, double riskReward, double successProbability) CalculateShortPositionQuality(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var scores = new List<double>();

            // 1. Trend alignment
            if (currentIndex >= 50)
            {
                var ema9 = CalculateEMA(bars, 9, currentIndex);
                var ema21 = CalculateEMA(bars, 21, currentIndex);
                var ema50 = CalculateEMA(bars, 50, currentIndex);

                if (ema9 < ema21 && ema21 < ema50) scores.Add(1.0);
                else if (ema9 < ema21) scores.Add(0.6);
                else scores.Add(0.2);
            }

            // 2. Momentum
            var rsi = CalculateRSI(bars, 14, currentIndex);
            if (rsi > 30 && rsi < 70) scores.Add(1.0);
            else if (rsi <= 30) scores.Add(0.5);
            else scores.Add(0.3);

            // 3. Volume confirmation
            var currentVol = (double)bars[currentIndex].TickVolume;
            var avgVol = 0.0;
            for (int i = currentIndex - 19; i <= currentIndex - 1; i++)
            {
                avgVol += bars[i].TickVolume;
            }
            avgVol /= 19;

            if (currentVol > avgVol * 1.2) scores.Add(1.0);
            else if (currentVol > avgVol) scores.Add(0.7);
            else scores.Add(0.4);

            // 4. Resistance nearby
            var resistanceDist = FindNearestResistance(bars, currentIndex);
            if (resistanceDist < 5.0) scores.Add(1.0);
            else if (resistanceDist < 10.0) scores.Add(0.7);
            else scores.Add(0.3);

            // 5. Volatility favorable
            var atr = CalculateATR(bars, 14, currentIndex);
            var avgAtr = 0.0;
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                avgAtr += CalculateATR(bars, 14, i);
            }
            avgAtr /= 20;

            if (atr < avgAtr * 1.5) scores.Add(1.0);
            else scores.Add(0.5);

            var quality = scores.Average();
            var entryScore = quality * (1.0 + (bars[currentIndex].DownVolume / (double)(bars[currentIndex].UpVolume + bars[currentIndex].DownVolume)));
            var riskReward = CalculateShortRiskReward(bars, currentIndex);
            var successProb = quality * 0.8;

            return (quality, entryScore / 2, riskReward, successProb);
        }

        private (double recommendedPos, double confidence) DeterminePositionRecommendation(
            (double quality, double entryScore, double riskReward, double successProbability) longQuality,
            (double quality, double entryScore, double riskReward, double successProbability) shortQuality)
        {
            var longScore = (longQuality.quality + longQuality.entryScore + longQuality.riskReward + longQuality.successProbability) / 4;
            var shortScore = (shortQuality.quality + shortQuality.entryScore + shortQuality.riskReward + shortQuality.successProbability) / 4;

            var diff = Math.Abs(longScore - shortScore);
            var confidence = Math.Min(1.0, diff * 2);

            if (longScore > shortScore && longScore > 0.6)
                return (1.0, confidence);
            else if (shortScore > longScore && shortScore > 0.6)
                return (-1.0, confidence);
            else
                return (0.0, 0.0);
        }

        private (double signal, double strength, double confirmation) DetectLongEntrySignal(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var signals = new List<double>();

            // Price pullback to support
            var support = FindNearestSupport(bars, currentIndex);
            if (support < 3.0) signals.Add(1.0);

            // RSI oversold recovery
            var rsi = CalculateRSI(bars, 14, currentIndex);
            if (rsi > 30 && rsi < 50)
            {
                var prevRsi = currentIndex >= 15 ? CalculateRSI(bars, 14, currentIndex - 1) : rsi;
                if (rsi > prevRsi) signals.Add(1.0);
            }

            // MACD crossover
            if (currentIndex >= 26)
            {
                var macdCurrent = CalculateMACD(bars, currentIndex);
                var macdPrev = CalculateMACD(bars, currentIndex - 1);
                if (macdCurrent > 0 && macdPrev <= 0) signals.Add(1.0);
            }

            // Volume surge
            var volRatio = bars[currentIndex].TickVolume / (bars[currentIndex - 1].TickVolume + 1.0);
            if (volRatio > 1.5) signals.Add(1.0);

            var signal = signals.Count > 0 ? 1.0 : 0.0;
            var strength = signals.Count > 0 ? signals.Average() : 0.0;
            var confirmation = (double)signals.Count / 4.0;

            return (signal, strength, confirmation);
        }

        private (double signal, double strength, double confirmation) DetectShortEntrySignal(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var signals = new List<double>();

            // Price rejection at resistance
            var resistance = FindNearestResistance(bars, currentIndex);
            if (resistance < 3.0) signals.Add(1.0);

            // RSI overbought reversal
            var rsi = CalculateRSI(bars, 14, currentIndex);
            if (rsi < 70 && rsi > 50)
            {
                var prevRsi = currentIndex >= 15 ? CalculateRSI(bars, 14, currentIndex - 1) : rsi;
                if (rsi < prevRsi) signals.Add(1.0);
            }

            // MACD crossunder
            if (currentIndex >= 26)
            {
                var macdCurrent = CalculateMACD(bars, currentIndex);
                var macdPrev = CalculateMACD(bars, currentIndex - 1);
                if (macdCurrent < 0 && macdPrev >= 0) signals.Add(1.0);
            }

            // Volume surge
            var volRatio = bars[currentIndex].TickVolume / (bars[currentIndex - 1].TickVolume + 1.0);
            if (volRatio > 1.5) signals.Add(1.0);

            var signal = signals.Count > 0 ? 1.0 : 0.0;
            var strength = signals.Count > 0 ? signals.Average() : 0.0;
            var confirmation = (double)signals.Count / 4.0;

            return (signal, strength, confirmation);
        }

        private double DetectLongExitSignal(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Exit when momentum fades or resistance hit
            var rsi = CalculateRSI(bars, 14, currentIndex);
            if (rsi > 70) return 1.0;

            var resistance = FindNearestResistance(bars, currentIndex);
            if (resistance < 2.0) return 1.0;

            return 0.0;
        }

        private double DetectShortExitSignal(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Exit when momentum fades or support hit
            var rsi = CalculateRSI(bars, 14, currentIndex);
            if (rsi < 30) return 1.0;

            var support = FindNearestSupport(bars, currentIndex);
            if (support < 2.0) return 1.0;

            return 0.0;
        }

        private (bool wouldActivate, double potentialProfit, double maxFavorable) AnalyzeTrailingStopLong(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 10) return (false, 0, 0);

            var entryPrice = (double)bars[currentIndex].Low; // Assume entry at low
            var maxProfit = 0.0;

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                var profit = ((double)bars[i].High - entryPrice) * 10000;
                maxProfit = Math.Max(maxProfit, profit);
            }

            var wouldActivate = maxProfit >= TRAILING_STOP_ACTIVATION_PIPS;
            var potentialProfit = Math.Max(0, maxProfit - TRAILING_STOP_DISTANCE_PIPS);

            return (wouldActivate, potentialProfit, maxProfit);
        }

        private (bool wouldActivate, double potentialProfit, double maxFavorable) AnalyzeTrailingStopShort(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 10) return (false, 0, 0);

            var entryPrice = (double)bars[currentIndex].High;
            var maxProfit = 0.0;

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                var profit = (entryPrice - (double)bars[i].Low) * 10000;
                maxProfit = Math.Max(maxProfit, profit);
            }

            var wouldActivate = maxProfit >= TRAILING_STOP_ACTIVATION_PIPS;
            var potentialProfit = Math.Max(0, maxProfit - TRAILING_STOP_DISTANCE_PIPS);

            return (wouldActivate, potentialProfit, maxProfit);
        }

        private (double downsideRisk, double upsidePotential, double asymmetry, double stopDistance) CalculateRiskMetrics(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var close = (double)bars[currentIndex].Close;
            var atr = CalculateATR(bars, 14, currentIndex);

            var support = FindNearestSupportLevel(bars, currentIndex);
            var resistance = FindNearestResistanceLevel(bars, currentIndex);

            var downsideRisk = (close - support) * 10000;
            var upsidePotential = (resistance - close) * 10000;
            var asymmetry = SafeDiv(upsidePotential, downsideRisk);
            var stopDistance = atr * 10000 * 2; // 2x ATR stop

            return (downsideRisk, upsidePotential, asymmetry, stopDistance);
        }

        private (double supportStrength, double resistanceStrength, double trendAlignment, double momentumAlignment) AnalyzeMarketStructure(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Support/Resistance strength based on touches
            var supportStrength = CalculateSupportStrength(bars, currentIndex);
            var resistanceStrength = CalculateResistanceStrength(bars, currentIndex);

            // Trend alignment
            var ema9 = CalculateEMA(bars, 9, currentIndex);
            var ema21 = CalculateEMA(bars, 21, currentIndex);
            var trendAlignment = SafeDiv(ema9 - ema21, ema21);

            // Momentum alignment
            var rsi = CalculateRSI(bars, 14, currentIndex);
            var momentumAlignment = (rsi - 50) / 50;

            return (supportStrength, resistanceStrength, trendAlignment, momentumAlignment);
        }

        private (double longEntry, double shortEntry) FindOptimalEntryLevels(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var close = (double)bars[currentIndex].Close;
            var atr = CalculateATR(bars, 14, currentIndex);

            // Long entry: slight pullback to support or EMA
            var ema9 = CalculateEMA(bars, 9, currentIndex);
            var longEntry = Math.Min(close - atr * 0.5, ema9);

            // Short entry: slight rally to resistance or EMA
            var shortEntry = Math.Max(close + atr * 0.5, ema9);

            return (longEntry, shortEntry);
        }

        private double EstimateHoldingDuration(IReadOnlyList<OhlcBar> bars, int currentIndex, bool isLong)
        {
            // Estimate bars until target hit (simplified)
            var atr = CalculateATR(bars, 14, currentIndex);
            var targetMove = TRAILING_STOP_ACTIVATION_PIPS * 0.0001;

            // Average bars per ATR move
            var recentMoves = 0.0;
            var barCount = 0;

            for (int i = currentIndex - 9; i < currentIndex; i++)
            {
                var move = Math.Abs((double)(bars[i + 1].Close - bars[i].Close));
                if (move > 0)
                {
                    recentMoves += move;
                    barCount++;
                }
            }

            var avgMovePerBar = barCount > 0 ? recentMoves / barCount : atr;
            return SafeDiv(targetMove, avgMovePerBar);
        }

        private (double longAlignment, double shortAlignment, double consensus) CalculateMultiTimeframeAlignment(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Check alignment across different EMA periods
            var ema9 = CalculateEMA(bars, 9, currentIndex);
            var ema21 = CalculateEMA(bars, 21, currentIndex);
            var ema50 = CalculateEMA(bars, 50, currentIndex);

            var longAlignment = 0.0;
            if (ema9 > ema21) longAlignment += 0.5;
            if (ema21 > ema50) longAlignment += 0.5;

            var shortAlignment = 0.0;
            if (ema9 < ema21) shortAlignment += 0.5;
            if (ema21 < ema50) shortAlignment += 0.5;

            var consensus = Math.Abs(longAlignment - shortAlignment);

            return (longAlignment, shortAlignment, consensus);
        }

        private (double longSize, double shortSize, double confidence) RecommendPositionSize(
            IReadOnlyList<OhlcBar> bars, int currentIndex,
            (double quality, double entryScore, double riskReward, double successProbability) longQuality,
            (double quality, double entryScore, double riskReward, double successProbability) shortQuality)
        {
            // Kelly Criterion inspired sizing
            var longSize = longQuality.quality * longQuality.successProbability;
            var shortSize = shortQuality.quality * shortQuality.successProbability;
            var confidence = Math.Max(longQuality.successProbability, shortQuality.successProbability);

            return (longSize, shortSize, confidence);
        }

        private double CalculateTradeExpectancy(IReadOnlyList<OhlcBar> bars, int currentIndex, bool isLong)
        {
            // E = (Win% * AvgWin) - (Loss% * AvgLoss)
            var winProb = EstimateWinProbability(bars, currentIndex, isLong);
            var avgWin = TRAILING_STOP_ACTIVATION_PIPS - TRAILING_STOP_DISTANCE_PIPS;
            var avgLoss = MAX_DRAWDOWN_PIPS;

            return (winProb * avgWin) - ((1 - winProb) * avgLoss);
        }

        private double CalculateSetupCorrelation(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Compare current setup to recent successful setups
            var currentRsi = CalculateRSI(bars, 14, currentIndex);
            var currentATR = CalculateATR(bars, 14, currentIndex);

            var similarities = new List<double>();

            foreach (var setup in _setupHistory.GetValues().Take(10))
            {
                // Simple similarity metric
                var similarity = 1.0 / (1.0 + Math.Abs(setup.ExpectedProfit - currentATR));
                similarities.Add(similarity);
            }

            return similarities.Count > 0 ? similarities.Average() : 0.5;
        }

        private double CalculateAdverseSelectionRisk(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Risk of being on wrong side
            var spread = (double)bars[currentIndex].AvgSpread;
            var atr = CalculateATR(bars, 14, currentIndex);

            // Higher spread relative to ATR = higher adverse selection
            return Math.Min(1.0, SafeDiv(spread * 10, atr));
        }

        private double EstimateSlippage(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Slippage estimate based on spread and volatility
            var spread = (double)bars[currentIndex].AvgSpread;
            var atr = CalculateATR(bars, 14, currentIndex);

            // During high volatility, slippage increases
            return spread * (1 + Math.Min(2.0, SafeDiv(atr, spread)));
        }

        private double EstimateWinProbability(IReadOnlyList<OhlcBar> bars, int currentIndex, bool isLong)
        {
            // Historical win probability based on similar market conditions
            var quality = isLong ?
                CalculateLongPositionQuality(bars, currentIndex).quality :
                CalculateShortPositionQuality(bars, currentIndex).quality;

            // Base probability + quality adjustment
            return 0.4 + (quality * 0.3); // 40-70% range
        }

        // Helper methods
        private double FindNearestSupport(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var support = FindNearestSupportLevel(bars, currentIndex);
            var close = (double)bars[currentIndex].Close;
            return (close - support) * 10000;
        }

        private double FindNearestResistance(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var resistance = FindNearestResistanceLevel(bars, currentIndex);
            var close = (double)bars[currentIndex].Close;
            return (resistance - close) * 10000;
        }

        private double FindNearestSupportLevel(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var close = (double)bars[currentIndex].Close;
            var lows = new List<double>();

            for (int i = Math.Max(0, currentIndex - 50); i < currentIndex; i++)
            {
                lows.Add((double)bars[i].Low);
            }

            var support = lows.Where(l => l < close).DefaultIfEmpty(close * 0.999).Max();
            return support;
        }

        private double FindNearestResistanceLevel(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var close = (double)bars[currentIndex].Close;
            var highs = new List<double>();

            for (int i = Math.Max(0, currentIndex - 50); i < currentIndex; i++)
            {
                highs.Add((double)bars[i].High);
            }

            var resistance = highs.Where(h => h > close).DefaultIfEmpty(close * 1.001).Min();
            return resistance;
        }

        private double CalculateSupportStrength(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var support = FindNearestSupportLevel(bars, currentIndex);
            var touches = 0;

            for (int i = Math.Max(0, currentIndex - 50); i < currentIndex; i++)
            {
                if (Math.Abs((double)bars[i].Low - support) < 0.0002) touches++;
            }

            return Math.Min(1.0, touches / 5.0);
        }

        private double CalculateResistanceStrength(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var resistance = FindNearestResistanceLevel(bars, currentIndex);
            var touches = 0;

            for (int i = Math.Max(0, currentIndex - 50); i < currentIndex; i++)
            {
                if (Math.Abs((double)bars[i].High - resistance) < 0.0002) touches++;
            }

            return Math.Min(1.0, touches / 5.0);
        }

        private double CalculateLongRiskReward(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var close = (double)bars[currentIndex].Close;
            var support = FindNearestSupportLevel(bars, currentIndex);
            var resistance = FindNearestResistanceLevel(bars, currentIndex);

            var risk = (close - support) * 10000;
            var reward = (resistance - close) * 10000;

            return SafeDiv(reward, risk);
        }

        private double CalculateShortRiskReward(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var close = (double)bars[currentIndex].Close;
            var support = FindNearestSupportLevel(bars, currentIndex);
            var resistance = FindNearestResistanceLevel(bars, currentIndex);

            var risk = (resistance - close) * 10000;
            var reward = (close - support) * 10000;

            return SafeDiv(reward, risk);
        }

        private double CalculateRSI(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 50;

            double gains = 0;
            double losses = 0;

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

        private double CalculateMACD(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 26) return 0;

            var ema12 = CalculateEMA(bars, 12, currentIndex);
            var ema26 = CalculateEMA(bars, 26, currentIndex);

            return ema12 - ema26;
        }

        public override void Reset()
        {
            _positionHistory.Clear();
            _setupHistory.Clear();
        }
    }
}