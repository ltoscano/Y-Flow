from __future__ import print_function
import httplib2
import sys

from apiclient import discovery

from get_credentials import get_credentials
from read_words import read_words


def main(argv):

    if len(argv) != 4:
        print("Usage: translate.py [source language] [target language] [txt filename]")
        return

    """
    Creates a Sheets API service object and prints the names and majors of
    students in a sample spreadsheet:
    https://docs.google.com/spreadsheets/d/17EJkhCG_EnPFVfg6x-ghlchG-ieA2suPPJDE-2pdtp4/edit
    """
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                    'version=v4')
    service = discovery.build('sheets', 'v4', http=http,
                              discoveryServiceUrl=discoveryUrl)

    spreadsheetId = '17EJkhCG_EnPFVfg6x-ghlchG-ieA2suPPJDE-2pdtp4'

    fname = argv[3]
    words_2_translate = read_words(fname)
    words_count = len(words_2_translate)

    original_language = argv[1]
    translation_language = argv[2]

    i = 1
    insertValues = []
    for w in words_2_translate:
        insertValues.append([
            w, '=GOOGLETRANSLATE(A{row_num}, \"{original_language}\", \"{translation_language}\")'.format(row_num=i, original_language=original_language, translation_language=translation_language)])
        i += 1

    body = {
        'values': insertValues
    }

    rangeName = 'Sheet1!A1:B{}'.format(len(insertValues))

    service.spreadsheets().values().update(
        spreadsheetId=spreadsheetId, range=rangeName,
        valueInputOption='USER_ENTERED', body=body).execute()

    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheetId, range=rangeName).execute()

    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        for row in values:
            print(row[0], '|', row[1])


if __name__ == '__main__':
    main(sys.argv)
