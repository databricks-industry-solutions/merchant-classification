import re


nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero 

nMNTH = r'(?:11|12|10|0?[1-9])' # month can be 1 to 12 with a leading zero

nYR = r'(?:(?:19|20)\d\d)'  # I've restricted the year to being in 20th or 21st century on the basis 

nDELIM = r'(?:[\/\-\._])?' 

NUM_DATE = f"""
    (?P<num_date>
        (?:^|\D) # new bit here
        (?:
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
        )
        (?:\D|$) # new bit here
    )"""

DAY = r"""
(?:
    # search 1st 2nd 3rd etc, or first second third
    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)
    |
    # or just a number
    (?:[0123]?\d)
)"""

MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'

YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""

DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'

YEAR_4D = r"""(?:[12]\d\d\d)"""

DATE_PATTERN = f"""(?P<wordy_date>
    # non word character or start of string
    (?:^|\W)
        (?:
            # match various combinations of year month and day 
            (?:
                # 4 digit year
                (?:{YEAR_4D}{DELIM})?
                    (?:
                    # Day - Month
                    (?:{DAY}{DELIM}{MONTH})
                    |
                    # Month - Day
                    (?:{MONTH}{DELIM}{DAY})
                    )
                # 2 or 4 digit year
                (?:{DELIM}{YEAR})?
            )
            |
            # Month - Year (2 or 3 digit)
            (?:{MONTH}{DELIM}{YEAR})
            # non delimited dates
            |
            (?:{DAY}{MONTH}{YEAR})
            |
            (?:{DAY}{MONTH}{YEAR_4D})
            |
            (?:xx{DELIM}xx{DELIM}{YEAR_4D})
        )
    # non-word character or end of string
    (?:$|\W)
)"""

TIME = r"""(?:
(?:
# first number should be 0 - 59 with optional leading zero.
[012345]?\d
# second number is the same following a colon or a dot or h
(:|\.|h)[012345]\d
)
# next we add our optional seconds number in the same format
(?::[012345]\d)?
# and finally add optional am or pm possibly with . and spaces
(?:\s*(?:a|p)\.?m\.?)?
)"""

COMBINED = f"""(?P<combined>
    (?:
        # time followed by date, or date followed by time
        {TIME}?{DATE_PATTERN}{TIME}?
        |
        # or as above but with the numeric version of the date
        {TIME}?{NUM_DATE}{TIME}?
    ) 
    # or a time on its own
    |
    (?:{TIME})
)"""

price_regex = "(((?:\\d+\.)*\\d+,\\d+)|(\\d+\.\\d+))(?:[/\\s]*)(?:(gbp|\%))"

date_pattern = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)