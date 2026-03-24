<?php

declare(strict_types=1);
session_start();
header('Content-Type: text/html; charset=UTF-8');

const ISS_BASE = 'https://iss.moex.com/iss';
const RF = 0.16;
const CACHE_TTL = 86400;

$INDEX_META = [
    'IMOEX' => [
        'name' => 'Индекс МосБиржи',
        'ui_label' => 'IMOEX',
        'aliases' => ['IMOEX'],
        'color' => '#c8a86b',
        'max' => 80,
    ],
    'RGBITR' => [
        'name' => 'Гос. облигации (ОФЗ), совокупный доход',
        'ui_label' => 'RGBITR',
        'aliases' => ['RGBITR'],
        'color' => '#64748b',
        'max' => 80,
    ],
    'RUGOLD' => [
        'name' => 'Индекс МосБиржи аффинированного золота',
        'ui_label' => 'RUGOLD',
        'aliases' => ['RUGOLD'],
        'color' => '#d4af37',
        'max' => 80,
    ],
    'RUCBTR' => [
        'name' => 'Корп. облигации, совокупный доход',
        'ui_label' => 'RUCBTR',
        'aliases' => ['RUCBTRNS'],
        'color' => '#94a3b8',
        'max' => 80,
    ],
    'MREFTR' => [
        'name' => 'Индекс МосБиржи фондов недвижимости полной доходности',
        'ui_label' => 'MREFTR',
        'aliases' => ['MREFTR'],
        'color' => '#2dd4bf',
        'max' => 40,
    ],
];

$BASE = ['IMOEX', 'RGBITR', 'RUGOLD', 'RUCBTR'];
$ALL = ['IMOEX', 'RGBITR', 'RUGOLD', 'RUCBTR', 'MREFTR'];

$BASELINE = [
    'IMOEX' => 40,
    'RGBITR' => 25,
    'RUGOLD' => 20,
    'RUCBTR' => 15,
    'MREFTR' => 0,
];

$MREF_PORTFOLIO = [
    'IMOEX' => 34,
    'RGBITR' => 21,
    'RUGOLD' => 17,
    'RUCBTR' => 13,
    'MREFTR' => 15,
];

$METRIC_DEFS = [
    [
        'label' => 'Годовая доходность',
        'key' => 'ret',
        'mult' => 100,
        'dec' => 1,
        'suf' => '%',
        'dir' => 1,
        'gt' => 12,
        'ot' => 7,
        'desc' => 'Среднегодовая взвешенная доходность портфеля',
    ],
    [
        'label' => 'Волатильность',
        'key' => 'vol',
        'mult' => 100,
        'dec' => 1,
        'suf' => '%',
        'dir' => -1,
        'gt' => 12,
        'ot' => 18,
        'desc' => 'Стандартное отклонение — мера нестабильности',
    ],
    [
        'label' => 'Коэф. Шарпа',
        'key' => 'shr',
        'mult' => 1,
        'dec' => 2,
        'suf' => '',
        'dir' => 1,
        'gt' => 0.3,
        'ot' => 0,
        'desc' => 'Доходность сверх ставки ЦБ (16%) на единицу риска',
    ],
    [
        'label' => 'Коэф. Сортино',
        'key' => 'sor',
        'mult' => 1,
        'dec' => 2,
        'suf' => '',
        'dir' => 1,
        'gt' => 0.4,
        'ot' => 0,
        'desc' => 'Шарп, штрафующий только нисходящую волатильность',
    ],
    [
        'label' => 'Макс. просадка',
        'key' => 'mdd',
        'mult' => 100,
        'dec' => 1,
        'suf' => '%',
        'dir' => -1,
        'gt' => -15,
        'ot' => -30,
        'desc' => 'Максимальное падение стоимости от исторического пика',
    ],
    [
        'label' => 'Коэф. Кальмара',
        'key' => 'cal',
        'mult' => 1,
        'dec' => 2,
        'suf' => '',
        'dir' => 1,
        'gt' => 0.5,
        'ot' => 0.2,
        'desc' => 'Доходность / |Макс. просадка| — качество риска/награды',
    ],
];

function h(string $value): string
{
    return htmlspecialchars($value, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function cache_dir(): string
{
    $dir = __DIR__ . DIRECTORY_SEPARATOR . 'cache';
    if (!is_dir($dir)) {
        mkdir($dir, 0775, true);
    }
    return $dir;
}

function init_state(array $all, array $baseline): void
{
    $expectedKeys = $all;
    sort($expectedKeys);

    if (!isset($_SESSION['weights']) || !is_array($_SESSION['weights'])) {
        $_SESSION['weights'] = $baseline;
    } else {
        $currentKeys = array_keys($_SESSION['weights']);
        sort($currentKeys);
        if ($currentKeys !== $expectedKeys) {
            $_SESSION['weights'] = $baseline;
        }
    }

    if (!isset($_SESSION['re_on'])) {
        $_SESSION['re_on'] = false;
    }

    if (!isset($_SESSION['prev_re_on'])) {
        $_SESSION['prev_re_on'] = false;
    }

    foreach ($all as $ticker) {
        $sliderKey = 'slider_' . $ticker;
        if (!isset($_SESSION[$sliderKey])) {
            $_SESSION[$sliderKey] = (int) ($_SESSION['weights'][$ticker] ?? 0);
        }
    }

    foreach (['slider_MCFTR', 'slider_IMOEXTR', 'slider_RGBI', 'slider_MREF'] as $oldKey) {
        if (isset($_SESSION[$oldKey])) {
            unset($_SESSION[$oldKey]);
        }
    }
}

function rebalance_for_toggle(bool $enableMref, array $all, array $baseline, array $mrefPortfolio): void
{
    $_SESSION['weights'] = $enableMref ? $mrefPortfolio : $baseline;

    foreach ($all as $ticker) {
        $_SESSION['slider_' . $ticker] = (int) $_SESSION['weights'][$ticker];
    }
}

function iss_get(string $url, ?array $params = null): array
{
    $params = $params ?? [];
    ksort($params);
    $cacheKey = sha1($url . '|' . http_build_query($params));
    $cacheFile = cache_dir() . DIRECTORY_SEPARATOR . $cacheKey . '.json';

    if (is_file($cacheFile) && (time() - filemtime($cacheFile) < CACHE_TTL)) {
        $cached = file_get_contents($cacheFile);
        if ($cached !== false) {
            $decoded = json_decode($cached, true);
            if (is_array($decoded)) {
                return $decoded;
            }
        }
    }

    $fullUrl = $url;
    if (!empty($params)) {
        $fullUrl .= '?' . http_build_query($params);
    }

    $ch = curl_init($fullUrl);
    if ($ch === false) {
        throw new RuntimeException('Не удалось инициализировать cURL.');
    }

    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_CONNECTTIMEOUT => 10,
        CURLOPT_TIMEOUT => 30,
        CURLOPT_HTTPGET => true,
        CURLOPT_USERAGENT => 'PortfolioRussianIndexesPHP/1.0',
    ]);

    $body = curl_exec($ch);
    if ($body === false) {
        $error = curl_error($ch);
        curl_close($ch);
        throw new RuntimeException('Ошибка запроса к ISS: ' . $error);
    }

    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($status >= 400) {
        throw new RuntimeException('ISS вернул HTTP ' . $status);
    }

    $decoded = json_decode($body, true);
    if (!is_array($decoded)) {
        throw new RuntimeException('ISS вернул некорректный JSON.');
    }

    file_put_contents($cacheFile, json_encode($decoded, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES));
    return $decoded;
}

function block_to_rows(array $block): array
{
    $columns = $block['columns'] ?? [];
    $data = $block['data'] ?? [];
    if (!is_array($columns) || !is_array($data)) {
        return [];
    }

    $rows = [];
    foreach ($data as $row) {
        if (is_array($row)) {
            $assoc = [];
            foreach ($columns as $idx => $column) {
                $assoc[(string) $column] = $row[$idx] ?? null;
            }
            $rows[] = $assoc;
        }
    }
    return $rows;
}

function resolve_board(string $secid): array
{
    $json = iss_get(
        ISS_BASE . '/securities/' . rawurlencode($secid) . '.json',
        ['iss.meta' => 'off', 'iss.only' => 'boards']
    );

    $boards = block_to_rows($json['boards'] ?? []);
    if ($boards === []) {
        throw new RuntimeException('Не найден блок boards для ' . $secid);
    }

    usort($boards, static function (array $a, array $b): int {
        return ((int) ($b['is_traded'] ?? 0)) <=> ((int) ($a['is_traded'] ?? 0));
    });

    $cand = array_values(array_filter($boards, static function (array $row): bool {
        return (string) ($row['engine'] ?? '') === 'stock';
    }));
    if ($cand === []) {
        $cand = $boards;
    }

    $pref = array_values(array_filter($cand, static function (array $row): bool {
        return (string) ($row['market'] ?? '') === 'index';
    }));

    $pick = $pref[0] ?? $cand[0] ?? null;
    if ($pick === null) {
        throw new RuntimeException('Не удалось определить board для ' . $secid);
    }

    return [
        (string) ($pick['engine'] ?? ''),
        (string) ($pick['market'] ?? ''),
        (string) ($pick['boardid'] ?? ''),
    ];
}

function load_index_candles_daily(string $secid, string $dateFrom, string $dateTo): array
{
    [$engine, $market, $board] = resolve_board($secid);
    $url = ISS_BASE . '/engines/' . rawurlencode($engine)
        . '/markets/' . rawurlencode($market)
        . '/boards/' . rawurlencode($board)
        . '/securities/' . rawurlencode($secid) . '/candles.json';

    $frames = [];
    $start = 0;

    while (true) {
        $json = iss_get($url, [
            'from' => $dateFrom,
            'till' => $dateTo,
            'interval' => 24,
            'start' => $start,
            'iss.meta' => 'off',
        ]);

        $candlesBlock = $json['candles'] ?? [];
        $part = block_to_rows($candlesBlock);

        if ($part === []) {
            break;
        }

        $frames = array_merge($frames, $part);
        $got = count($part);
        if ($got < 100) {
            break;
        }

        $start += $got;
    }

    if ($frames === []) {
        return [];
    }

    $daily = [];
    foreach ($frames as $row) {
        $begin = (string) ($row['begin'] ?? '');
        $close = $row['close'] ?? null;
        $tradedate = substr($begin, 0, 10);
        if ($tradedate === '' || !is_numeric((string) $close)) {
            continue;
        }
        $daily[$tradedate] = [
            'secid' => $secid,
            'tradedate' => $tradedate,
            'close' => (float) $close,
        ];
    }

    ksort($daily);
    return array_values($daily);
}

function load_one_index(string $logicalKey, string $dateFrom, string $dateTo, array $indexMeta): array
{
    $aliases = $indexMeta[$logicalKey]['aliases'] ?? [];
    $lastError = 'неизвестная ошибка';

    foreach ($aliases as $secid) {
        try {
            $data = load_index_candles_daily((string) $secid, $dateFrom, $dateTo);
            if ($data !== []) {
                return [$data, (string) $secid];
            }
        } catch (Throwable $e) {
            $lastError = $e->getMessage();
        }
    }

    throw new RuntimeException('Не удалось загрузить ' . $logicalKey . '. Последняя ошибка: ' . $lastError);
}

function load_all_index_data(string $dateFrom, string $dateTo, array $all, array $indexMeta): array
{
    $rawMap = [];
    $resolvedMap = [];
    $allDatesMap = [];

    foreach ($all as $key) {
        [$data, $resolved] = load_one_index($key, $dateFrom, $dateTo, $indexMeta);
        $rawMap[$key] = $data;
        $resolvedMap[$key] = $resolved;
        foreach ($data as $row) {
            $allDatesMap[$row['tradedate']] = true;
        }
    }

    $allDates = array_keys($allDatesMap);
    sort($allDates);

    $seriesMap = [];
    foreach ($all as $key) {
        $seriesMap[$key] = [];
        foreach ($rawMap[$key] as $row) {
            $seriesMap[$key][$row['tradedate']] = (float) $row['close'];
        }
    }

    $lastSeen = array_fill_keys($all, null);
    $merged = [];
    foreach ($allDates as $tradeDate) {
        $row = ['tradedate' => $tradeDate];
        foreach ($all as $key) {
            if (array_key_exists($tradeDate, $seriesMap[$key])) {
                $lastSeen[$key] = $seriesMap[$key][$tradeDate];
            }
            $row[$key] = $lastSeen[$key];
        }
        $merged[] = $row;
    }

    return [$merged, $resolvedMap, $rawMap];
}

function asset_stats_from_raw(array $rows): ?array
{
    if ($rows === []) {
        return null;
    }

    usort($rows, static function (array $a, array $b): int {
        return strcmp((string) $a['tradedate'], (string) $b['tradedate']);
    });

    $prices = array_map(static fn(array $row): float => (float) $row['close'], $rows);
    if (count($prices) < 2) {
        return null;
    }

    $returns = [];
    for ($i = 1, $n = count($prices); $i < $n; $i++) {
        if ($prices[$i - 1] == 0.0) {
            continue;
        }
        $returns[] = ($prices[$i] / $prices[$i - 1]) - 1.0;
    }

    if ($returns === []) {
        return null;
    }

    $first = $prices[0];
    $last = $prices[count($prices) - 1];
    if ($first <= 0.0 || $last <= 0.0) {
        return null;
    }

    $cagr = pow($last / $first, 252 / count($returns)) - 1;
    $vol = stddev_pop($returns) * sqrt(252);

    return [$cagr, $vol];
}

function prepare_window(array $prices, array $activeKeys, array $all): array
{
    $out = [];
    foreach ($prices as $row) {
        $valid = true;
        foreach ($activeKeys as $key) {
            if (!isset($row[$key]) || $row[$key] === null || !is_numeric((string) $row[$key])) {
                $valid = false;
                break;
            }
        }
        if (!$valid) {
            continue;
        }

        $prepared = ['tradedate' => (string) $row['tradedate']];
        foreach ($all as $key) {
            $prepared[$key] = isset($row[$key]) && is_numeric((string) $row[$key]) ? (float) $row[$key] : null;
        }
        $out[] = $prepared;
    }

    return $out;
}

function portfolio_daily_returns(array $priceWindow, array $weightsPct, array $all): array
{
    $used = [];
    foreach ($all as $key) {
        if ((int) ($weightsPct[$key] ?? 0) > 0) {
            $used[] = $key;
        }
    }

    if ($used === [] || count($priceWindow) < 2) {
        return [];
    }

    $returns = [];
    for ($i = 1, $n = count($priceWindow); $i < $n; $i++) {
        $rowRet = 0.0;
        $valid = true;
        foreach ($used as $key) {
            $prev = $priceWindow[$i - 1][$key] ?? null;
            $curr = $priceWindow[$i][$key] ?? null;
            if (!is_numeric((string) $prev) || !is_numeric((string) $curr) || (float) $prev == 0.0) {
                $valid = false;
                break;
            }
            $assetRet = ((float) $curr / (float) $prev) - 1.0;
            $rowRet += $assetRet * (((float) $weightsPct[$key]) / 100.0);
        }
        if ($valid) {
            $returns[] = $rowRet;
        }
    }

    return $returns;
}

function calc_portfolio_metrics(array $priceWindow, array $weightsPct, array $all): array
{
    $pr = portfolio_daily_returns($priceWindow, $weightsPct, $all);
    if ($pr === []) {
        return ['ret' => 0.0, 'vol' => 0.0, 'shr' => 0.0, 'sor' => 0.0, 'mdd' => 0.0, 'cal' => 0.0];
    }

    $nav = [];
    $cum = 1.0;
    foreach ($pr as $ret) {
        $cum *= (1.0 + $ret);
        $nav[] = $cum;
    }

    $annReturn = pow($nav[count($nav) - 1], 252 / count($pr)) - 1;
    $annVol = stddev_pop($pr) * sqrt(252);

    $downsideSquares = [];
    foreach ($pr as $ret) {
        $down = min($ret, 0.0);
        $downsideSquares[] = $down * $down;
    }
    $downsideDev = sqrt(mean($downsideSquares)) * sqrt(252);

    $sharpe = $annVol != 0.0 ? (($annReturn - RF) / $annVol) : 0.0;
    $sortino = $downsideDev != 0.0 ? (($annReturn - RF) / $downsideDev) : 0.0;

    $runningMax = null;
    $mdd = 0.0;
    foreach ($nav as $value) {
        $runningMax = $runningMax === null ? $value : max($runningMax, $value);
        $drawdown = ($runningMax != 0.0) ? ($value / $runningMax - 1.0) : 0.0;
        $mdd = min($mdd, $drawdown);
    }

    $calmar = $mdd != 0.0 ? ($annReturn / abs($mdd)) : 0.0;

    return [
        'ret' => $annReturn,
        'vol' => $annVol,
        'shr' => $sharpe,
        'sor' => $sortino,
        'mdd' => $mdd,
        'cal' => $calmar,
    ];
}

function metric_class(int $direction, float $value, float $goodThr, float $okThr): string
{
    if ($direction === 1) {
        if ($value >= $goodThr) {
            return 'good';
        }
        if ($value >= $okThr) {
            return 'neutral';
        }
        return 'bad';
    }

    if ($value <= $goodThr) {
        return 'good';
    }
    if ($value <= $okThr) {
        return 'neutral';
    }
    return 'bad';
}

function format_delta(string $metricKey, float $currentValue, float $baseValue, int $dec, string $suffix): array
{
    $diff = $currentValue - $baseValue;
    $icon = $diff >= 0 ? '▲' : '▼';
    $sign = $diff >= 0 ? '+' : '';

    if ($metricKey === 'vol') {
        $improved = $currentValue < $baseValue;
    } elseif ($metricKey === 'mdd') {
        $improved = abs($currentValue) < abs($baseValue);
    } else {
        $improved = $currentValue > $baseValue;
    }

    $css = $improved ? 'pos' : 'neg';
    $text = sprintf('%s %s%s%s vs базовый', $icon, $sign, number_format($diff, $dec, '.', ''), $suffix);
    return [$text, $css];
}

function mean(array $values): float
{
    if ($values === []) {
        return 0.0;
    }
    return array_sum($values) / count($values);
}

function stddev_pop(array $values): float
{
    if ($values === []) {
        return 0.0;
    }
    $avg = mean($values);
    $sum = 0.0;
    foreach ($values as $value) {
        $sum += ($value - $avg) ** 2;
    }
    return sqrt($sum / count($values));
}

init_state($ALL, $BASELINE);

$defaultFrom = '2025-01-01';
$defaultTo = date('Y-m-d');

if (!isset($_SESSION['date_from'])) {
    $_SESSION['date_from'] = $defaultFrom;
}
if (!isset($_SESSION['date_to'])) {
    $_SESSION['date_to'] = $defaultTo;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $_SESSION['date_from'] = isset($_POST['date_from']) ? (string) $_POST['date_from'] : $_SESSION['date_from'];
    $_SESSION['date_to'] = isset($_POST['date_to']) ? (string) $_POST['date_to'] : $_SESSION['date_to'];
    $_SESSION['re_on'] = isset($_POST['re_on']);

    if ((bool) $_SESSION['re_on'] !== (bool) $_SESSION['prev_re_on']) {
        rebalance_for_toggle((bool) $_SESSION['re_on'], $ALL, $BASELINE, $MREF_PORTFOLIO);
        $_SESSION['prev_re_on'] = (bool) $_SESSION['re_on'];
    } else {
        foreach ($ALL as $ticker) {
            $field = 'slider_' . $ticker;
            if (isset($_POST[$field]) && is_numeric((string) $_POST[$field])) {
                $value = (int) $_POST[$field];
                $maxAllowed = (int) ($INDEX_META[$ticker]['max'] ?? 100);
                $value = max(0, min($maxAllowed, $value));
                $_SESSION[$field] = $value;
                $_SESSION['weights'][$ticker] = $value;
            }
        }
    }
}

if (!$_SESSION['re_on']) {
    $_SESSION['weights']['MREFTR'] = 0;
    $_SESSION['slider_MREFTR'] = 0;
}

$dFrom = (string) $_SESSION['date_from'];
$dTo = (string) $_SESSION['date_to'];
$reOn = (bool) $_SESSION['re_on'];
$activeTickers = $reOn ? $ALL : $BASE;

$errors = [];
$loadError = null;
$prices = [];
$resolvedMap = [];
$rawMap = [];
$assetStats = [];
$priceWindow = [];
$currentMetrics = null;
$baselineMetrics = null;
$actualStart = null;
$actualEnd = null;
$currentTotal = 0;

if ($dFrom >= $dTo) {
    $errors[] = 'Дата начала должна быть раньше даты окончания.';
} else {
    try {
        [$prices, $resolvedMap, $rawMap] = load_all_index_data($dFrom, $dTo, $ALL, $INDEX_META);
    } catch (Throwable $e) {
        $loadError = $e->getMessage();
    }
}

if ($loadError === null && $errors === []) {
    foreach ($ALL as $key) {
        $assetStats[$key] = asset_stats_from_raw($rawMap[$key] ?? []);
    }

    $currentTotal = 0;
    foreach ($activeTickers as $ticker) {
        $currentTotal += (int) ($_SESSION['weights'][$ticker] ?? 0);
    }

    $priceWindow = prepare_window($prices, $activeTickers, $ALL);
    if (count($priceWindow) < 3) {
        $errors[] = 'Недостаточно общих исторических данных для расчета метрик.';
    } else {
        $actualStart = $priceWindow[0]['tradedate'];
        $actualEnd = $priceWindow[count($priceWindow) - 1]['tradedate'];
    }

    if ($errors === [] && abs($currentTotal - 100) <= 1) {
        $currentMetrics = calc_portfolio_metrics($priceWindow, $_SESSION['weights'], $ALL);
        $baselineMetrics = calc_portfolio_metrics($priceWindow, $BASELINE, $ALL);
    }
}
?>
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Портфель российских индексов</title>
    <style>
        :root {
            --bg: #0a0c10;
            --panel: #111419;
            --panel-2: #121825;
            --panel-3: #0f1520;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --muted-2: #64748b;
            --border: #1e2530;
            --gold: #c8a86b;
            --gold-light: #e8c97a;
            --green: #4ade80;
            --red: #f87171;
            --teal: #2dd4bf;
        }

        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
        }
        .page {
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 20px 40px;
        }
        .topbar {
            display: grid;
            grid-template-columns: 3.2fr 2fr;
            gap: 20px;
            align-items: start;
            margin-bottom: 16px;
        }
        .title h1 {
            margin: 0 0 6px;
            font-size: 2.1rem;
            font-weight: 800;
        }
        .title h1 span { color: var(--gold-light); }
        .caption {
            color: var(--muted);
            font-size: 0.95rem;
        }
        .toggle-card {
            background: linear-gradient(135deg, var(--panel-2), var(--panel-3));
            border: 1px solid rgba(200,168,107,0.55);
            border-radius: 22px;
            padding: 14px 18px 12px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
        }
        .toggle-card-grid {
            display: grid;
            grid-template-columns: 4.2fr 1.1fr;
            gap: 12px;
            align-items: center;
        }
        .toggle-title {
            color: var(--text);
            font-size: 1.02rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 8px;
            line-height: 1.1;
        }
        .toggle-name {
            color: #e5e7eb;
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 4px;
            line-height: 1.2;
        }
        .toggle-sub {
            color: var(--muted);
            font-size: 0.80rem;
            line-height: 1.35;
        }
        .toggle-wrap {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .toggle-wrap input[type="checkbox"] {
            width: 24px;
            height: 24px;
            cursor: pointer;
            accent-color: var(--teal);
        }
        details.period {
            background: #111419;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 18px;
        }
        details.period summary {
            cursor: pointer;
            font-weight: 700;
            color: var(--text);
        }
        .period-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 14px;
        }
        .field label {
            display: block;
            color: #fff;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .field input[type="date"] {
            width: 100%;
            background: #0f1520;
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
        }
        .section-title {
            font-size: 0.82rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 24px 0 12px;
            font-weight: 700;
        }
        .asset-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
        }
        .asset-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px 14px 12px;
            min-height: 185px;
        }
        .asset-ticker {
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }
        .asset-name {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 10px;
            min-height: 40px;
        }
        .slider-row {
            margin: 12px 0 6px;
        }
        .slider-row input[type="range"] {
            width: 100%;
            accent-color: var(--gold);
        }
        .weight-value {
            font-weight: 700;
            margin-bottom: 8px;
        }
        .asset-stat {
            color: #cbd5e1;
            font-size: 0.78rem;
            margin-top: 8px;
            line-height: 1.45;
        }
        .asset-stat-label { color: #cbd5e1; }
        .asset-stat-pos { color: var(--green); font-weight: 600; }
        .asset-stat-light { color: #cbd5e1; }
        .banner {
            background: linear-gradient(135deg, rgba(200,168,107,0.08), rgba(200,168,107,0.02));
            border: 1px solid rgba(200,168,107,0.2);
            border-radius: 12px;
            padding: 14px 18px;
            color: var(--gold);
            margin: 8px 0 22px;
        }
        .alert {
            border-radius: 12px;
            padding: 14px 16px;
            margin: 10px 0 18px;
            border: 1px solid rgba(248,113,113,0.28);
            color: #fecaca;
            background: rgba(127,29,29,0.22);
        }
        .meta-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin: 10px 0 0;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(200px, 1fr));
            gap: 14px;
        }
        .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px 16px;
            min-height: 160px;
        }
        .metric-label {
            font-size: 0.72rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 8px;
        }
        .good { color: var(--green); }
        .neutral { color: var(--gold-light); }
        .bad { color: var(--red); }
        .metric-delta {
            font-size: 0.80rem;
            margin-bottom: 8px;
            min-height: 1.2em;
        }
        .pos { color: var(--green); }
        .neg { color: var(--red); }
        .zero { color: var(--muted-2); }
        .metric-desc {
            font-size: 0.78rem;
            color: var(--muted-2);
            line-height: 1.45;
        }
        .insight {
            background: linear-gradient(135deg, rgba(45,212,191,0.06), rgba(45,212,191,0.02));
            border: 1px solid rgba(45,212,191,0.18);
            border-radius: 14px;
            padding: 20px;
            margin-top: 14px;
            margin-bottom: 22px;
        }
        .insight h3 {
            color: var(--teal);
            margin-top: 0;
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            line-height: 1.5;
        }
        .footnote {
            border-top: 1px solid var(--border);
            padding-top: 16px;
            margin-top: 16px;
            color: var(--muted-2);
            font-size: 0.78rem;
            line-height: 1.6;
        }
        .submit-row {
            display: flex;
            justify-content: flex-end;
            margin-top: 14px;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            appearance: none;
            border: 1px solid rgba(200,168,107,0.35);
            background: linear-gradient(135deg, rgba(200,168,107,0.14), rgba(200,168,107,0.06));
            color: var(--text);
            border-radius: 10px;
            padding: 10px 14px;
            font-weight: 700;
            cursor: pointer;
        }
        .hint {
            color: var(--muted-2);
            font-size: 0.76rem;
            margin-top: 6px;
        }
        @media (max-width: 960px) {
            .topbar,
            .toggle-card-grid,
            .period-grid,
            .metric-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
<div class="page">
    <form method="post" id="portfolio-form">
        <details class="period">
            <summary>Период расчета</summary>
            <div class="period-grid">
                <div class="field">
                    <label for="date_from">Дата начала</label>
                    <input type="date" id="date_from" name="date_from" value="<?= h($dFrom) ?>">
                </div>
                <div class="field">
                    <label for="date_to">Дата окончания</label>
                    <input type="date" id="date_to" name="date_to" value="<?= h($dTo) ?>">
                </div>
            </div>
        </details>

        <div class="topbar">
            <div class="title">
                <h1>Портфель <span>Российских Индексов</span></h1>
                <div class="caption">Интерактивный анализ · Метрики риска и доходности</div>
            </div>
            <div class="toggle-card">
                <div class="toggle-card-grid">
                    <div>
                        <div class="toggle-title">Добавить недвижимость</div>
                        <div class="toggle-name">MREFTR • Складская недвижимость</div>
                        <div class="toggle-sub">Индекс МосБиржи фондов недвижимости полной доходности</div>
                    </div>
                    <div class="toggle-wrap">
                        <input type="checkbox" id="re_on" name="re_on" <?= $reOn ? 'checked' : '' ?>>
                    </div>
                </div>
            </div>
        </div>

        <?php if ($reOn): ?>
            <div class="banner"><strong>Складская недвижимость добавлена.</strong> Низкая корреляция с акциями — портфель становится устойчивее без потери доходности.</div>
        <?php endif; ?>

        <?php foreach ($errors as $error): ?>
            <div class="alert"><?= h($error) ?></div>
        <?php endforeach; ?>

        <?php if ($loadError !== null): ?>
            <div class="alert">Ошибка загрузки данных: <?= h($loadError) ?></div>
        <?php endif; ?>

        <div class="section-title">Распределение активов</div>
        <div class="asset-grid">
            <?php foreach ($activeTickers as $ticker): ?>
                <?php
                    $meta = $INDEX_META[$ticker];
                    $sliderKey = 'slider_' . $ticker;
                    $sliderValue = (int) ($_SESSION[$sliderKey] ?? 0);
                    $stat = $assetStats[$ticker] ?? null;
                ?>
                <div class="asset-card">
                    <div class="asset-ticker" style="color: <?= h($meta['color']) ?>"><?= h($meta['ui_label']) ?></div>
                    <div class="asset-name"><?= h($meta['name']) ?></div>
                    <div class="slider-row">
                        <input
                            type="range"
                            name="<?= h($sliderKey) ?>"
                            min="0"
                            max="<?= (int) $meta['max'] ?>"
                            step="1"
                            value="<?= $sliderValue ?>"
                            data-output="out_<?= h($ticker) ?>"
                        >
                    </div>
                    <div class="weight-value"><span id="out_<?= h($ticker) ?>"><?= $sliderValue ?></span>%</div>
                    <?php if (is_array($stat)): ?>
                        <div class="asset-stat">
                            <span class="asset-stat-label">Доходность:</span>
                            <span class="asset-stat-pos"><?= number_format($stat[0] * 100, 1, '.', '') ?>% / год</span>
                            ·
                            <span class="asset-stat-label">Вол:</span>
                            <span class="asset-stat-light"><?= number_format($stat[1] * 100, 1, '.', '') ?>%</span>
                        </div>
                    <?php else: ?>
                        <div class="asset-stat">Недостаточно данных</div>
                    <?php endif; ?>
                </div>
            <?php endforeach; ?>
        </div>


    <?php if ($loadError === null && $errors === []): ?>
        <?php if (abs($currentTotal - 100) > 1): ?>
            <div class="alert">Сумма весов не равна 100%. Текущая сумма: <?= (int) $currentTotal ?>%</div>
        <?php elseif (is_array($currentMetrics) && is_array($baselineMetrics)): ?>
            <div class="section-title">Портфельные метрики</div>
            <div class="metric-grid">
                <?php foreach ($METRIC_DEFS as $md): ?>
                    <?php
                        $raw = (float) $currentMetrics[$md['key']];
                        $value = $raw * (float) $md['mult'];
                        if ($value < 0) {
                            $cssClass = 'bad';
                        } else {
                            $cssClass = metric_class((int) $md['dir'], $value, (float) $md['gt'], (float) $md['ot']);
                        }
                        $deltaHtml = '<div class="metric-delta zero"></div>';
                        if ($reOn) {
                            $baseValue = (float) $baselineMetrics[$md['key']] * (float) $md['mult'];
                            [$deltaText, $deltaCss] = format_delta((string) $md['key'], $value, $baseValue, (int) $md['dec'], (string) $md['suf']);
                            $deltaHtml = '<div class="metric-delta ' . h($deltaCss) . '">' . h($deltaText) . '</div>';
                        }
                    ?>
                    <div class="metric-card">
                        <div class="metric-label"><?= h($md['label']) ?></div>
                        <div class="metric-value <?= h($cssClass) ?>"><?= number_format($value, (int) $md['dec'], '.', '') . h($md['suf']) ?></div>
                        <?= $deltaHtml ?>
                        <div class="metric-desc"><?= h($md['desc']) ?></div>
                    </div>
                <?php endforeach; ?>
            </div>
        <?php endif; ?>
    <?php endif; ?>

    <?php if ($reOn): ?>
        <div class="insight">
            <h3>Почему склады улучшают портфель?</h3>
            <div class="insight-grid">
                <div>📦 MREFTR — индекс МосБиржи, отражающий динамику фондов складской и индустриальной недвижимости. Включает несколько ЗПИФов, инвестирующих в объекты логистики и склады класса A/B.</div>
                <div>📈 Фонды в составе индекса обеспечивают регулярные рентные выплаты — источник дохода даже при падении фондового рынка.</div>
                <div>🛡 Добавление слабо коррелированного актива снижает волатильность портфеля без потери доходности — эффект диверсификации.</div>
                <div>⚖️ В терминах портфельной теории некоррелированный актив может улучшать отношение риска к доходности и коэффициент Шарпа.</div>
                <div>🏗 Рост e-commerce и 3PL-логистики поддерживает спрос на складскую недвижимость как отдельный инвестиционный сегмент.</div>
                <div>🔒 Недвижимость часто рассматривается как частичная инфляционная защита при индексации ставок аренды и росте стоимости замещения.</div>
            </div>
        </div>
    <?php endif; ?>

    <div class="footnote">
        Источник данных: ISS Московской биржи. Метрики рассчитываются по историческим дневным значениям индексов за выбранный период. IMOEX, RGBITR, RUGOLD, RUCBTRNS — официальные индексы МосБиржи. MREFTR — индекс МосБиржи фондов недвижимости полной доходности. Безрисковая ставка для коэффициентов Шарпа и Сортино принята равной 16% (ключевая ставка ЦБ РФ). Расчеты носят аналитический характер и не являются инвестиционной рекомендацией.
    </div>
</div>
<script>
    document.querySelectorAll('input[type="range"][data-output]').forEach(function (slider) {
        var output = document.getElementById(slider.dataset.output);
        var update = function () {
            if (output) output.textContent = slider.value;
        };
        slider.addEventListener('input', update);
        update();
    });

    var form = document.getElementById('portfolio-form');
    var toggle = document.getElementById('re_on');
    var dateFrom = document.getElementById('date_from');
    var dateTo = document.getElementById('date_to');

    [toggle, dateFrom, dateTo].forEach(function (el) {
        if (!el) return;
        el.addEventListener('change', function () {
            form.submit();
        });
    });
</script>
</body>
</html>
