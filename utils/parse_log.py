import apache_log_parser
import pandas as pd


def parse_log_line(line):
    line_parser = apache_log_parser.make_parser(
        '%h %l %u %t "%r" %>s %b "%{Referer}i" "%{User-Agent}i"'
    )
    try:
        data = line_parser(line)
        return {
            "request": data.get("request_method") + " " + data.get("request_url"),
            "referer": data.get("request_header_referer") or None,
            "user_agent": data.get("request_header_user_agent"),
        }
    except Exception as e:
        return {"error": str(e), "log": line}


def parse_logs_into_df(df, log_column="log"):
    parsed = df[log_column].apply(parse_log_line)
    parsed_df = pd.DataFrame(parsed.tolist())
    return pd.concat([df, parsed_df], axis=1)
