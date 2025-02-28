daily_data_query = """
            SELECT tb_max.datetime AS datetime, tb_max.tickersymbol AS symbol,
            tb_max.price AS max,
            tb_min.price AS min,
            tb_close.price AS "close",
            tb_open.price AS "open",
            tb_total.quantity AS total_quantity
            
            FROM (
                SELECT *
                FROM quote.max m
            ) tb_max
            INNER JOIN (
                SELECT *
                FROM quote.min m
            ) tb_min
            ON tb_max.tickersymbol = tb_min.tickersymbol
            AND tb_max.datetime = tb_min.datetime
            INNER JOIN (
                SELECT *
                FROM quote.close m
            ) tb_close
            ON tb_min.tickersymbol = tb_close.tickersymbol
            AND tb_min.datetime = tb_close.datetime
            INNER JOIN (
                SELECT *
                FROM quote.open m
            ) tb_open
            ON tb_open.tickersymbol = tb_close.tickersymbol
            AND tb_open.datetime = tb_close.datetime
            INNER JOIN (
                SELECT TO_CHAR(datetime , 'YYYY-MM-DD') AS datetime , m.tickersymbol, max(m.quantity) AS quantity
                FROM quote.total m
                WHERE 
                    (m.tickersymbol = 'VN30F2310' AND m.datetime BETWEEN '2023-09-30' AND '2023-10-21')
                OR 
                    (m.tickersymbol = 'VN30F2311' AND m.datetime BETWEEN '2023-10-21' AND '2023-11-17')
                OR
                    (m.tickersymbol = 'VN30F2312' AND m.datetime BETWEEN '2023-11-17' AND '2023-12-22')
                OR
                    (m.tickersymbol = 'VN30F2401' AND m.datetime BETWEEN '2023-12-22' AND '2024-01-01')
                GROUP BY TO_CHAR(datetime , 'YYYY-MM-DD'), m.tickersymbol
            ) tb_total
            ON tb_open.tickersymbol = tb_total.tickersymbol
            AND tb_open.datetime = to_date(tb_total.datetime, 'YYYY-MM-DD')
            WHERE 
                (tb_max.tickersymbol = 'VN30F2310' AND tb_max.datetime BETWEEN '2023-09-30' AND '2023-10-21')
            OR 
                (tb_max.tickersymbol = 'VN30F2311' AND tb_max.datetime BETWEEN '2023-10-21' AND '2023-11-17')
            OR
                (tb_max.tickersymbol = 'VN30F2312' AND tb_max.datetime BETWEEN '2023-11-17' AND '2023-12-22')
            OR
                (tb_max.tickersymbol = 'VN30F2401' AND tb_max.datetime BETWEEN '2023-12-22' AND '2024-01-01')
"""
