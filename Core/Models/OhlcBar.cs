namespace ForexFeatureGenerator.Core.Models
{
    public record OhlcBar
    {
        public DateTime Timestamp { get; init; }
        public TimeSpan Timeframe { get; init; }
        public decimal Open { get; init; }
        public decimal High { get; init; }
        public decimal Low { get; init; }
        public decimal Close { get; init; }
        public decimal TypicalPrice => (High + Low + Close) / 3;
        public int TickVolume { get; init; }
        public decimal UpVolume { get; init; } // NEW - sum of up tick volumes
        public decimal DownVolume { get; init; } // NEW - sum of down tick volumes
        public decimal AvgSpread { get; init; }
        public decimal MaxSpread { get; init; }
        public decimal MinSpread { get; init; }
    }
}
