daily_data_query = """
SELECT 
    tb_max.datetime AS datetime, 
    tb_max.tickersymbol AS symbol,
    tb_max.price AS max,
    tb_min.price AS min,
    tb_close.price AS close,
    tb_open.price AS open

FROM (
    SELECT * FROM quote.max m
) tb_max

INNER JOIN (
    SELECT * FROM quote.min m
) tb_min
ON tb_max.tickersymbol = tb_min.tickersymbol
AND tb_max.datetime = tb_min.datetime

INNER JOIN (
    SELECT * FROM quote.close m
) tb_close
ON tb_min.tickersymbol = tb_close.tickersymbol
AND tb_min.datetime = tb_close.datetime

INNER JOIN (
    SELECT * FROM quote.open m
) tb_open
ON tb_open.tickersymbol = tb_close.tickersymbol
AND tb_open.datetime = tb_close.datetime

WHERE 
    (tb_max.tickersymbol = 'VN30F2101' AND tb_max.datetime BETWEEN '2021-01-04' AND '2021-01-14')
OR 
    (tb_max.tickersymbol = 'VN30F2102' AND tb_max.datetime BETWEEN '2021-01-15' AND '2021-02-18')
OR 
    (tb_max.tickersymbol = 'VN30F2103' AND tb_max.datetime BETWEEN '2021-02-19' AND '2021-03-18')
OR
    (tb_max.tickersymbol = 'VN30F2104' AND tb_max.datetime BETWEEN '2021-03-19' AND '2021-04-15')
OR
    (tb_max.tickersymbol = 'VN30F2105' AND tb_max.datetime BETWEEN '2021-04-16' AND '2021-05-20')
OR
    (tb_max.tickersymbol = 'VN30F2106' AND tb_max.datetime BETWEEN '2021-05-21' AND '2021-06-17')
OR
    (tb_max.tickersymbol = 'VN30F2107' AND tb_max.datetime BETWEEN '2021-06-18' AND '2021-07-15')
OR
    (tb_max.tickersymbol = 'VN30F2108' AND tb_max.datetime BETWEEN '2021-07-16' AND '2021-08-19')
OR
    (tb_max.tickersymbol = 'VN30F2109' AND tb_max.datetime BETWEEN '2021-08-20' AND '2021-09-16')
OR
    (tb_max.tickersymbol = 'VN30F2110' AND tb_max.datetime BETWEEN '2021-09-17' AND '2021-10-21')
OR
    (tb_max.tickersymbol = 'VN30F2111' AND tb_max.datetime BETWEEN '2021-10-22' AND '2021-11-18')
OR
    (tb_max.tickersymbol = 'VN30F2112' AND tb_max.datetime BETWEEN '2021-11-19' AND '2021-12-16')
OR
    (tb_max.tickersymbol = 'VN30F2201' AND tb_max.datetime BETWEEN '2021-12-17' AND '2022-01-20')
OR
    (tb_max.tickersymbol = 'VN30F2202' AND tb_max.datetime BETWEEN '2022-01-21' AND '2022-02-17')
OR
    (tb_max.tickersymbol = 'VN30F2203' AND tb_max.datetime BETWEEN '2022-02-18' AND '2022-03-17')
OR
    (tb_max.tickersymbol = 'VN30F2204' AND tb_max.datetime BETWEEN '2022-03-18' AND '2022-04-21')
OR
    (tb_max.tickersymbol = 'VN30F2205' AND tb_max.datetime BETWEEN '2022-04-22' AND '2022-05-19')
OR
    (tb_max.tickersymbol = 'VN30F2206' AND tb_max.datetime BETWEEN '2022-05-20' AND '2022-06-16')
OR
    (tb_max.tickersymbol = 'VN30F2207' AND tb_max.datetime BETWEEN '2022-06-17' AND '2022-07-21')
OR
    (tb_max.tickersymbol = 'VN30F2208' AND tb_max.datetime BETWEEN '2022-07-22' AND '2022-08-18')
OR
    (tb_max.tickersymbol = 'VN30F2209' AND tb_max.datetime BETWEEN '2022-08-19' AND '2022-09-15')
OR
    (tb_max.tickersymbol = 'VN30F2210' AND tb_max.datetime BETWEEN '2022-09-16' AND '2022-10-20')
OR
    (tb_max.tickersymbol = 'VN30F2211' AND tb_max.datetime BETWEEN '2022-10-21' AND '2022-11-17')
OR
    (tb_max.tickersymbol = 'VN30F2212' AND tb_max.datetime BETWEEN '2022-11-18' AND '2022-12-15')
OR
    (tb_max.tickersymbol = 'VN30F2301' AND tb_max.datetime BETWEEN '2022-12-16' AND '2023-01-19')
OR
    (tb_max.tickersymbol = 'VN30F2302' AND tb_max.datetime BETWEEN '2023-01-20' AND '2023-02-16')
OR
    (tb_max.tickersymbol = 'VN30F2303' AND tb_max.datetime BETWEEN '2023-02-17' AND '2023-03-16')
OR
    (tb_max.tickersymbol = 'VN30F2304' AND tb_max.datetime BETWEEN '2023-03-17' AND '2023-04-20')
OR
    (tb_max.tickersymbol = 'VN30F2305' AND tb_max.datetime BETWEEN '2023-04-21' AND '2023-05-18')
OR
    (tb_max.tickersymbol = 'VN30F2306' AND tb_max.datetime BETWEEN '2023-05-19' AND '2023-06-15')
OR
    (tb_max.tickersymbol = 'VN30F2307' AND tb_max.datetime BETWEEN '2023-06-16' AND '2023-07-20')
OR
    (tb_max.tickersymbol = 'VN30F2308' AND tb_max.datetime BETWEEN '2023-07-21' AND '2023-08-17')
OR
    (tb_max.tickersymbol = 'VN30F2309' AND tb_max.datetime BETWEEN '2023-08-18' AND '2023-09-21')
OR
    (tb_max.tickersymbol = 'VN30F2310' AND tb_max.datetime BETWEEN '2023-09-22' AND '2023-10-19')
OR
    (tb_max.tickersymbol = 'VN30F2311' AND tb_max.datetime BETWEEN '2023-10-20' AND '2023-11-16')
OR
    (tb_max.tickersymbol = 'VN30F2312' AND tb_max.datetime BETWEEN '2023-11-17' AND '2023-12-21')
OR
    (tb_max.tickersymbol = 'VN30F2401' AND tb_max.datetime BETWEEN '2023-12-22' AND '2024-01-18')
OR
    (tb_max.tickersymbol = 'VN30F2402' AND tb_max.datetime BETWEEN '2024-01-19' AND '2024-02-15')
OR
    (tb_max.tickersymbol = 'VN30F2403' AND tb_max.datetime BETWEEN '2024-02-16' AND '2024-03-21')
OR
    (tb_max.tickersymbol = 'VN30F2501' AND tb_max.datetime BETWEEN '2024-12-20' AND '2025-02-25')
ORDER BY tb_max.datetime ASC;
"""
