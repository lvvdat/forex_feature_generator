using ForexFeatureGenerator.Core.Models;

namespace ForexFeatureGenerator.Label
{
    // Label Generation Classes
    public class LabelGenerationConfig
    {
        public double StopLossPips { get; set; }  // If 0 or less, SL is inferred from spread & DistancePips
        public double TriggerPips { get; set; }
        public double DistancePips { get; set; }
        public int MaxFutureTicks { get; set; }
        public double MinConfidenceThreshold { get; set; }
        public double MinScoreThreshold { get; set; }
    }

    public class LabelResult
    {
        public int Label { get; set; }
        public double Confidence { get; set; }
        public double LongProfitPips { get; set; }
        public double ShortProfitPips { get; set; }
        public double MaxAdverseExcursion { get; set; }
        public double MaxFavorableExcursion { get; set; }
        public int TimeToTarget { get; set; }
        public double RiskRewardRatio { get; set; }
        public double QualityScore { get; set; }
    }

    // Integrated Label Generator
    public static class LabelGenerator
    {
        private const double DEFAULT_TAKE_PROFIT_MULTIPLIER = 3.0;
        private const double MAX_TIME_LIMIT_TICKS = 600;

        /// <summary>
        /// Generate label using trailing stop logic and a determined Stop Loss.
        /// Stop Loss selection:
        /// - If config.StopLossPips > 0, use it.
        /// - Else, infer SL from current spread and DistancePips with a minimum floor.
        /// </summary>
        public static LabelResult GenerateLabel(
            LabelGenerationConfig config,
            TickData currentTick,
            List<TickData> futureTicks)
        {
            if (futureTicks == null || futureTicks.Count < 10)
                return CreateNeutralResult();

            int ticksToAnalyze = Math.Min(futureTicks.Count, config.MaxFutureTicks);
            var analysisWindow = futureTicks.Take(ticksToAnalyze).ToList();

            // --- Determine Stop Loss (in pips) ---
            // NOTE: Uses 0.0001 as pip size. If you trade JPY or fractional pips,
            // add config.PipSize and replace PIP below accordingly.
            const double PIP = 0.0001;
            const double DEFAULT_MIN_SL_PIPS = 5.0;     // floor to avoid too-tight SL
            const double DEFAULT_SPREAD_MULT = 3.0;     // spread multiplier when inferring SL

            double spreadPips = (double)(currentTick.Ask - currentTick.Bid) / PIP;

            // Prefer explicit config, otherwise infer from spread & DistancePips
            double stopLossPips =
                (config.StopLossPips > 0.0)
                    ? config.StopLossPips
                    : Math.Max(
                        DEFAULT_MIN_SL_PIPS,
                        Math.Max(config.DistancePips, spreadPips * DEFAULT_SPREAD_MULT)
                      );

            // --- Simulate both directions with Stop Loss ---
            var longResult = SimulateTrailingStop(
                currentTick,
                analysisWindow,
                config.TriggerPips,
                config.DistancePips,
                stopLossPips,
                isLong: true);

            var shortResult = SimulateTrailingStop(
                currentTick,
                analysisWindow,
                config.TriggerPips,
                config.DistancePips,
                stopLossPips,
                isLong: false);

            return DetermineLabel(longResult, shortResult, config);
        }


        /// <summary>
        /// Simulates a trade outcome with activation-based trailing stop, fixed Stop Loss, and time limit.
        /// Exit priority per tick: StopLoss -> TakeProfit -> TrailingStop -> TimeLimit
        /// MFE/MAE tracked in price units and converted to pips at the end.
        /// </summary>
        private static TrailingStopResult SimulateTrailingStop(
            TickData entryTick,
            List<TickData> futureTicks,
            double activationPips,
            double distancePips,
            double stopLossPips,      // NEW: hard Stop Loss in pips (0 or less => disabled)
            bool isLong)
        {
            // NOTE: This uses 0.0001 as pip value. If you support JPY or fractional pips,
            // make this instrument-aware (e.g., pass in pipSize).
            const double PIP = 0.0001;

            double entryPrice = (double)(isLong ? entryTick.Ask : entryTick.Bid);
            double activationDistance = activationPips * PIP;
            double trailDistance = distancePips * PIP;
            double stopLossDistance = Math.Max(0.0, stopLossPips) * PIP; // <=0 disables SL
            double takeProfitDistance = activationDistance * DEFAULT_TAKE_PROFIT_MULTIPLIER;

            // Compute absolute Stop Loss price if enabled
            bool stopLossEnabled = stopLossDistance > 0.0;
            double stopLossPrice = stopLossEnabled
                ? (isLong ? entryPrice - stopLossDistance : entryPrice + stopLossDistance)
                : 0.0;

            bool trailingActivated = false;
            double trailingStop = 0.0;
            double maxFavorableExcursion = 0.0;
            double maxAdverseExcursion = 0.0;
            double exitPrice = 0.0;
            int exitTick = -1;
            string exitReason = "TimeLimit";

            for (int i = 0; i < futureTicks.Count; i++)
            {
                var tick = futureTicks[i];

                // Use same side as your original for P&L: long -> Bid, short -> Ask
                double currentPrice = (double)(isLong ? tick.Bid : tick.Ask);
                double priceMove = isLong ? (currentPrice - entryPrice)
                                             : (entryPrice - currentPrice);

                // Track MFE/MAE in price units
                if (priceMove >= 0)
                    maxFavorableExcursion = Math.Max(maxFavorableExcursion, priceMove);
                else
                    maxAdverseExcursion = Math.Max(maxAdverseExcursion, Math.Abs(priceMove));

                // --- Exit checks (priority order) ---

                // 1) Hard Stop Loss (if enabled): exit as soon as price crosses SL level
                if (stopLossEnabled)
                {
                    bool hitSL = isLong ? (currentPrice <= stopLossPrice)
                                        : (currentPrice >= stopLossPrice);
                    if (hitSL)
                    {
                        exitPrice = stopLossPrice; // assume fill at stop level
                        exitTick = i;
                        exitReason = "StopLoss";
                        break;
                    }
                }

                // 2) Take Profit (based on activation distance * multiplier from entry)
                if (priceMove >= takeProfitDistance)
                {
                    exitPrice = currentPrice;
                    exitTick = i;
                    exitReason = "TakeProfit";
                    break;
                }

                // 3) Trailing stop logic (activation first, then trail)
                if (!trailingActivated)
                {
                    if (priceMove >= activationDistance)
                    {
                        trailingActivated = true;
                        trailingStop = isLong
                            ? currentPrice - trailDistance
                            : currentPrice + trailDistance;
                    }
                }
                else
                {
                    if (isLong)
                    {
                        double newStop = currentPrice - trailDistance;
                        trailingStop = Math.Max(trailingStop, newStop);

                        if (currentPrice <= trailingStop)
                        {
                            exitPrice = trailingStop;
                            exitTick = i;
                            exitReason = "TrailingStop";
                            break;
                        }
                    }
                    else
                    {
                        double newStop = currentPrice + trailDistance;
                        trailingStop = Math.Min(trailingStop, newStop);

                        if (currentPrice >= trailingStop)
                        {
                            exitPrice = trailingStop;
                            exitTick = i;
                            exitReason = "TrailingStop";
                            break;
                        }
                    }
                }

                // 4) Time limit
                if (i >= MAX_TIME_LIMIT_TICKS)
                {
                    exitPrice = currentPrice;
                    exitTick = i;
                    exitReason = "TimeLimit";
                    break;
                }
            }

            // If no exit triggered, close at last tick
            if (exitTick < 0)
            {
                var lastTick = futureTicks.Last();
                exitPrice = (double)(isLong ? lastTick.Bid : lastTick.Ask);
                exitTick = futureTicks.Count - 1;
                exitReason = "TimeLimit";
            }

            // Convert P&L and excursions to pips
            double profitPips = (isLong ? (exitPrice - entryPrice) : (entryPrice - exitPrice)) / PIP;

            return new TrailingStopResult
            {
                ProfitPips = profitPips,
                MaxFavorableExcursionPips = maxFavorableExcursion / PIP,
                MaxAdverseExcursionPips = maxAdverseExcursion / PIP,
                TimeToExit = exitTick,
                ExitReason = exitReason,
                TrailingActivated = trailingActivated
            };
        }

        private static LabelResult DetermineLabel(
            TrailingStopResult longResult,
            TrailingStopResult shortResult,
            LabelGenerationConfig config)
        {
            double longQuality = CalculateQualityScore(longResult);
            double shortQuality = CalculateQualityScore(shortResult);
            double confidence = Math.Abs(longQuality - shortQuality);

            int label = 0;
            if (confidence >= config.MinConfidenceThreshold)
            {
                if (longQuality > shortQuality && longQuality >= config.MinScoreThreshold)
                    label = 1;
                else if (shortQuality > longQuality && shortQuality >= config.MinScoreThreshold)
                    label = -1;
            }

            return new LabelResult
            {
                Label = label,
                Confidence = Math.Min(1.0, confidence),
                LongProfitPips = longResult.ProfitPips,
                ShortProfitPips = shortResult.ProfitPips,
                MaxAdverseExcursion = Math.Max(longResult.MaxAdverseExcursionPips, shortResult.MaxAdverseExcursionPips),
                MaxFavorableExcursion = Math.Max(longResult.MaxFavorableExcursionPips, shortResult.MaxFavorableExcursionPips),
                TimeToTarget = label == 1 ? longResult.TimeToExit : label == -1 ? shortResult.TimeToExit : 0,
                QualityScore = Math.Max(longQuality, shortQuality),
                RiskRewardRatio = CalculateRiskReward(label == 1 ? longResult : shortResult)
            };
        }

        private static double CalculateQualityScore(TrailingStopResult result)
        {
            if (!result.TrailingActivated)
                return 0;

            double profitScore = Math.Min(1, Math.Max(0, result.ProfitPips / 10.0));
            double riskScore = result.MaxAdverseExcursionPips > 0 ?
                Math.Min(1, Math.Max(0, 1 - result.MaxAdverseExcursionPips / 10.0)) : 1;
            double timeScore = Math.Min(1, Math.Max(0, 1 - result.TimeToExit / MAX_TIME_LIMIT_TICKS));

            return (profitScore * 0.5 + riskScore * 0.3 + timeScore * 0.2);
        }

        private static double CalculateRiskReward(TrailingStopResult result)
        {
            if (result.MaxAdverseExcursionPips <= 0)
                return result.ProfitPips > 0 ? 10.0 : 0;

            return result.ProfitPips / result.MaxAdverseExcursionPips;
        }

        private static LabelResult CreateNeutralResult() => new LabelResult
        {
            Label = 0,
            Confidence = 0,
            QualityScore = 0,
            TimeToTarget = 0,
            RiskRewardRatio = 0
        };

        private class TrailingStopResult
        {
            public double ProfitPips { get; init; }
            public double MaxFavorableExcursionPips { get; init; }
            public double MaxAdverseExcursionPips { get; init; }
            public int TimeToExit { get; init; }
            public string ExitReason { get; init; } = "Unknown";
            public bool TrailingActivated { get; init; }
        }
    }
}