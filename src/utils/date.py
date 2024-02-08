import datetime

def string_to_datetime(string):
  split = string.split('-')
  year = int(split(0))
  month = int(split(1))
  day = int(split(2))

  return datetime.datetime(year=year, month=month, day=day)