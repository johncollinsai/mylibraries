import pyodbc

from .decorators import timed_print


class Connection(object):
    CNXN = None
    CURSOR = None

    def __init__(self, driver='SQL Server Native Client 11.0', database='norfolk_006228_0434_WP'):
        self.connect(driver=driver, database=database)

    @timed_print
    def connect(self, driver='', database=''):
        self.CNXN = pyodbc.connect(f'Driver={driver};Server=FEDAUK-SQL01;Database={database};Trusted_Connection=yes;')

        self.CURSOR = self.CNXN.cursor()

    @timed_print
    def query(self, query):
        if self.CURSOR is None:
            self.connect()

        self.CURSOR.execute(query)

        return [column[0] for column in self.CURSOR.description], [list(x) for x in self.CURSOR.fetchall()]


if __name__ == '__main__':
    pass
