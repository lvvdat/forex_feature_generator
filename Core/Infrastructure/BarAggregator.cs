using ForexFeatureGenerator.Core.Models;

namespace ForexFeatureGenerator.Core.Infrastructure
{
    public interface IBarAggregator
    {
        TimeSpan Timeframe { get; }
        void AddTick(TickData tick);
        OhlcBar? GetCompletedBar();
        OhlcBar? GetCurrentBar();
        IReadOnlyList<OhlcBar> GetHistoricalBars(int count);
        int HistoricalBarCount { get; }
    }

    public class BarAggregator : IBarAggregator
    {
        private readonly RollingWindow<OhlcBar> _historicalBars;
        private OhlcBar? _currentBar;
        private readonly List<TickData> _currentTicks = new();
        private OhlcBar? _lastCompletedBar;
        private decimal _lastMidPrice;
        private int _upTicks;
        private int _downTicks;
        private decimal _upVolume;
        private decimal _downVolume;

        public TimeSpan Timeframe { get; }
        public int HistoricalBarCount => _historicalBars.Count;

        public BarAggregator(TimeSpan timeframe, int historySize = 500)
        {
            Timeframe = timeframe;
            _historicalBars = new RollingWindow<OhlcBar>(historySize);
        }

        public void AddTick(TickData tick)
        {
            var barTime = GetBarTimestamp(tick.Timestamp);

            // Check if we need to complete the previous bar
            if (_currentBar != null && barTime > _currentBar.Timestamp)
            {
                // Finalize current bar with tick statistics
                _currentBar = _currentBar with
                {
                    UpTicks = _upTicks,
                    DownTicks = _downTicks,
                    UpVolume = _upVolume,
                    DownVolume = _downVolume
                };

                _historicalBars.Add(_currentBar);
                _lastCompletedBar = _currentBar;

                // Reset for new bar
                _currentBar = null;
                _currentTicks.Clear();
                _upTicks = 0;
                _downTicks = 0;
                _upVolume = 0;
                _downVolume = 0;
            }

            // Determine tick direction
            if (_lastMidPrice > 0)
            {
                if (tick.MidPrice > _lastMidPrice)
                {
                    _upTicks++;
                    _upVolume += 1; // In forex, we use tick count as volume proxy
                }
                else if (tick.MidPrice < _lastMidPrice)
                {
                    _downTicks++;
                    _downVolume += 1;
                }
            }
            _lastMidPrice = tick.MidPrice;

            // Start new bar or update current
            if (_currentBar == null)
            {
                _currentBar = new OhlcBar
                {
                    Timestamp = barTime,
                    Timeframe = Timeframe,
                    Open = tick.MidPrice,
                    High = tick.MidPrice,
                    Low = tick.MidPrice,
                    Close = tick.MidPrice,
                    TickVolume = 1,
                    AvgSpread = tick.Spread,
                    MaxSpread = tick.Spread,
                    MinSpread = tick.Spread
                };
            }
            else
            {
                _currentBar = _currentBar with
                {
                    High = Math.Max(_currentBar.High, tick.MidPrice),
                    Low = Math.Min(_currentBar.Low, tick.MidPrice),
                    Close = tick.MidPrice,
                    TickVolume = _currentBar.TickVolume + 1,
                    MaxSpread = Math.Max(_currentBar.MaxSpread, tick.Spread),
                    MinSpread = Math.Min(_currentBar.MinSpread, tick.Spread)
                };
            }

            _currentTicks.Add(tick);

            // Recalculate average spread
            if (_currentTicks.Count > 0)
            {
                _currentBar = _currentBar with
                {
                    AvgSpread = _currentTicks.Average(t => t.Spread)
                };
            }
        }

        public OhlcBar? GetCompletedBar()
        {
            var bar = _lastCompletedBar;
            _lastCompletedBar = null;
            return bar;
        }

        public OhlcBar? GetCurrentBar() => _currentBar;

        public IReadOnlyList<OhlcBar> GetHistoricalBars(int count)
        {
            var result = new List<OhlcBar>();
            int actualCount = Math.Min(count, _historicalBars.Count);

            for (int i = 0; i < actualCount; i++)
            {
                result.Add(_historicalBars[i]); // [0] is most recent
            }

            return result;
        }

        private DateTime GetBarTimestamp(DateTime timestamp)
        {
            var ticks = timestamp.Ticks;
            var timeframeTicks = Timeframe.Ticks;
            return new DateTime((ticks / timeframeTicks) * timeframeTicks, DateTimeKind.Utc);
        }
    }
}
