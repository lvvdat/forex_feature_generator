using System.Text;
using System.Globalization;
using System.Runtime.CompilerServices;

using ForexFeatureGenerator.Core.Models;

namespace ForexFeatureGenerator.Pipeline
{
    public static class TickLoader
    {
        // Stream ticks lazily, one line at a time
        public static async IAsyncEnumerable<TickData> ReadTicksAsync(
            string path,
            bool hasHeader = true,
            int progressEvery = 10_000,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"⚠️ Tick data file not found: {path}");

            // Configure async file IO with a generous buffer
            using var fs = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 1 << 20, // 1 MiB buffer
                options: FileOptions.Asynchronous | FileOptions.SequentialScan);

            using var reader = new StreamReader(fs, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 1 << 20);

            string? line;
            long lineNo = 0;
            long yielded = 0;

            if (hasHeader)
            {
                // Read and drop header
                await reader.ReadLineAsync().ConfigureAwait(false);
                lineNo++;
            }

            while ((line = await reader.ReadLineAsync().ConfigureAwait(false)) != null)
            {
                cancellationToken.ThrowIfCancellationRequested();
                lineNo++;

                // Skip blank or comment lines
                if (string.IsNullOrWhiteSpace(line) || line[0] == '#')
                    continue;

                if (TryParseTick(line.AsSpan(), out var tick))
                {
                    yield return tick!;
                    yielded++;

                    if (progressEvery > 0 && (yielded % progressEvery) == 0)
                    {
                        Log($"\r loaded {yielded:N0} ticks (line {lineNo:N0})");
                    }
                }
                else
                {
                    Log($"\n  ⚠️ Error parsing line {lineNo}", ConsoleColor.Yellow);
                }
            }

            Log($"\n  ✓ Loaded {yielded:N0} ticks");
        }

        // Convenience: load all ticks into a List<TickData>
        public static async Task<List<TickData>> LoadTickDataAsync(
            string path,
            bool hasHeader = true,
            int progressEvery = 10_000,
            CancellationToken cancellationToken = default)
        {
            Log("Loading tick data...");
            var list = new List<TickData>(capacity: 64_000);

            await foreach (var tick in ReadTicksAsync(path, hasHeader, progressEvery, cancellationToken)
                               .WithCancellation(cancellationToken)
                               .ConfigureAwait(false))
            {
                list.Add(tick);
            }

            return list;
        }

        // Fast, allocation-light CSV split for 3 columns: Timestamp,Bid,Ask
        private static bool TryParseTick(ReadOnlySpan<char> line, out TickData? tick)
        {
            tick = null;

            // Find first comma
            int i1 = line.IndexOf(',');
            if (i1 <= 0) return false;

            // Find second comma
            var rest = line.Slice(i1 + 1);
            int i2 = rest.IndexOf(',');
            if (i2 <= 0) return false;

            var tsSpan = line.Slice(0, i1).Trim();
            var bidSpan = rest.Slice(0, i2).Trim();
            var askSpan = rest.Slice(i2 + 1).Trim();

            if (!DateTime.TryParse(tsSpan, out var ts)) return false;
            if (!decimal.TryParse(bidSpan, NumberStyles.Float, CultureInfo.InvariantCulture, out var bid)) return false;
            if (!decimal.TryParse(askSpan, NumberStyles.Float, CultureInfo.InvariantCulture, out var ask)) return false;

            tick = new TickData { Timestamp = ts, Bid = bid, Ask = ask };
            return true;
        }

        // Replace with your logger; kept compatible with your existing calls
        private static void Log(string message, ConsoleColor color = ConsoleColor.Gray)
        {
            var prev = Console.ForegroundColor;
            Console.ForegroundColor = color;
            Console.Write(message);
            Console.ForegroundColor = prev;
        }
    }
}
