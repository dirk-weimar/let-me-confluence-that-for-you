import re
import tiktoken
import pandas as pd
import html2text
import streamlit as st
import time

from atlassian import Confluence
from bs4 import BeautifulSoup
from typing import List
from math import ceil
from io import StringIO

# ------------------ Config ------------------ #
confluence_url       = st.secrets['CONFLUENCE_URL']
confluence_username  = st.secrets['CONFLUENCE_USERNAME']
confluence_api_token = st.secrets['CONFLUENCE_API_TOKEN']

# Config for splitting large pages
max_tokens_per_page = 2600
max_characters_per_page = max_tokens_per_page * 3.3     # One word consists of 3.3 tokens on average
min_characters_per_page = max_characters_per_page / 3
max_rows_per_table = 20
marker = '\n##'


# ------------- Shared variables ------------- #
from module.shared import \
  max_num_tokens, \
  tokenizer_encoding_name, \
  embedding_model


# ------------- Shared functions ------------- #
from module.shared import \
  get_file_name_for_space, \
  create_embeddings


# ------------------ Helper ------------------ #
def get_num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def replace_structured_macros(html: str) -> str:
    soup = BeautifulSoup(html, features = 'html.parser')

    for tag in soup.find_all('ac:structured-macro'):

        if tag.get('ac:name') == 'code':
            code = soup.new_tag('code')
            code.string = tag.get_text()
            tag.replace_with(code)

    return soup.prettify(formatter = None)

def split_table(table_df):

    result_dfs = []

    # Split into chunks
    runs = ceil(len(table_df) / max_rows_per_table)

    for i in range(0, len(table_df), max_rows_per_table):
        chunk = table_df[i:i + max_rows_per_table]
        result_dfs.append(chunk)

    return result_dfs

def replace_table(match: re.Match):

    table_html      = match.group(0)
    table_text      = ''
    table_html_io   = StringIO(table_html)
    pandas_tables   = pd.read_html(table_html_io, header=0)

    try:
        table_df = pandas_tables[0]

        # Split big tables
        if len(table_df) > max_rows_per_table:
            table_dfs = split_table(table_df)

            for table_df in table_dfs:
                # Insert markdown headline before table als marker
                # which will be used to split large pages into smaller chunks
                table_text += '\n' +  marker + table_df.to_markdown(tablefmt = "markdown", index = False)

        else:
            table_text += marker + ' Tabelle:\n' + table_df.to_markdown(tablefmt = "jira", index = False)

        # Replace multiple blanks
        table_text = re.sub(r' {2,}', ' ', table_text, flags = re.DOTALL)

        return table_text


    except IndexError:
        return ''

def split_string_by_markers(input_string: str, marker: str, min_chunk_size: int, max_chunk_size: int) -> List:

    result= []
    current_chunk = ''

    splits = input_string.split(marker)

    for split in splits:

        # Füge Split zu aktuellem Chunk hinzu, wenn wir dadurch innerhalb der Obergrenze bleiben
        if len(current_chunk) + len(split) + len(marker) <= max_chunk_size:
            current_chunk += split + marker

        # Obergrenze würde durch Hinzufügen des Splitz gesprengt
        else:
            # Wenn vor Hinzufügen Mindestgrenze erreicht ist, beende aktuellen Chunk und
            # starte mit neuem Chunk mit aktuellem Split darin
            if len(current_chunk) >= min_chunk_size:
                result.append(current_chunk[:-3])
                current_chunk = split + marker

            # Wenn du dich entscheiden musst zwischen nicht erreichter Mindestgröße
            # und gesprengter Obergrenze, opfere im Zweifel die Obergrenze
            else:
                result.append(current_chunk[:-3])
                current_chunk += split + marker
                current_chunk = ''

    if current_chunk:
        result.append(current_chunk[:-3])

    return result

# ----------------- Functions ---------------- #
def connect_to_confluence() -> Confluence:
    url = confluence_url
    username = confluence_username
    api_token = confluence_api_token

    confluence = Confluence(url = url, username = username, password = api_token, cloud = True)

    return confluence

def get_confluence_spaces():
    confluence  = connect_to_confluence()
    spaces = confluence.get_all_spaces(start=0, limit=500, expand=None)
    spaces = spaces['results']
    result = []

    for space in spaces:
        entry = (space['key'], space['name'])
        result.append(entry)

    return result

def get_confluence_pages(space: str) -> list:

    confluence  = connect_to_confluence()

    # There is a limit of how many pages we can retrieve one at a time.
    # So we retrieve 100 at a time and loop until we know we retrieved all of them.
    keep_going = True
    start = 0
    limit = 100

    pages = []

    while keep_going:
        results = confluence.get_all_pages_from_space(\
          space, \
          start = start, \
          limit = 100, \
          status = None, \
          expand = 'body.storage', \
          content_type = 'page'\
        )

        pages.extend(results)

        if len(results) < limit:
            keep_going = False
        else:
            start = start + limit

    return pages

def filter_pages(pages: list) -> list:

    date_pattern = r'\d{4}-\d{2}-\d{2}'

    # Exclude pages without content
    condition1 = lambda page: len(page['body']['storage']['value']) > 0

    # Exclude meeting notes (incl. retros) containing a date like "02-23-2023"
    condition2 = lambda page: not re.match(date_pattern, page['title'])

    # ToDo: Exclude pages with "archiv" in path

    allowed_pages = [page for page in pages if condition1(page) and condition2(page)]

    return allowed_pages

def split_large_pages(pages: List) -> List:

    split_pages = []

    # page[0] = space
    # page[1] = title
    # page[2] = content
    # page[3] = link
    # page[4] = num_tokens

    for page in pages:

        if page[4] > max_tokens_per_page:

            content_list = split_string_by_markers(page[2], marker, min_characters_per_page, max_characters_per_page)

            i = 0

            for content_part in content_list:
                i += 1
                page_part = (\
                  page[0],\
                  page[1] + ' - Teil ' + str(i),\
                  content_part, \
                  page[3], \
                  get_num_tokens_from_string(content_part, tokenizer_encoding_name)\
                )

                if page_part[4] <= max_tokens_per_page:
                    split_pages.append(page_part)

                else:
                    # ToDO: Hard split
                    log_info = \
                      'skipped: ' + page_part[1] + ': ' \
                      + str(page_part[4]) + ' tokens after splitting (max. ' \
                      + str(max_tokens_per_page) + ')'

                    st.markdown(f'`{log_info}`')


        else:
            split_pages.append(page)

    return split_pages

def transform_html_to_text(html):

    # html2text does not understand confluence's macro-tags
    # Convert them to corresponding standard tags
    if '<ac:structured-macro' in html:
        html = replace_structured_macros(html)

    # Convert HTML to text
    # html2text does not excell in rendering tables, so bypass them for now
    text_maker = html2text.HTML2Text()
    text_maker.bypass_tables = True
    text_maker.body_width = 500
    text = text_maker.handle(html)

    # Convert tables
    pattern = r'<table>.*?</table>'
    matches = re.findall(pattern, text, flags = re.DOTALL)
    text = re.sub(pattern, replace_table, text, flags = re.DOTALL)

    # Remove newlines containing only blanks
    text = re.sub(r'^$\n', '', text, flags = re.MULTILINE)

    # Remove more than two newlines in a row
    text = re.sub(r'\n\s+', '\n\n', text, flags = re.MULTILINE)

    return text

def collect_data_from_confluence(space: str) -> list:

    # Get pages from Confluence space
    pages = get_confluence_pages(space)

    # Filter unneccessary and confidential pages
    pages = filter_pages(pages)

    # List of pages to be returned
    pages_data = []

    # Transform page content to readable text for AI
    for page in pages:

        # Debug single page html
        # if page['id'] != '3413748':
        #     continue

        id      = page['id']
        title   = page['title']
        link    = confluence_url + 'wiki/spaces/' + space + '/pages/' + page['id']
        body    = transform_html_to_text(page['body']['storage']['value'])

        # Merge title and body because that will be the context provided to the AI
        # so we need to count the tokens of both
        page_content = '\n*' + title + '*\n' + body # markdown for <h1>

        # Count number of tokens
        num_tokens = get_num_tokens_from_string(page_content, tokenizer_encoding_name)

        # Add to list
        pages_data += [(space, title, page_content, link, num_tokens)]

    return(pages_data)


# ------------------- Main ------------------- #
def write_csv(confluence_spaces: list, file_name: str) -> pd.DataFrame:

    for space in confluence_spaces:

        percent_complete = 0
        progress_text = 'Lade Informationen aus dem Confluence Space ' + space + ' ...'
        progress_bar = st.progress(percent_complete, text = progress_text)

        # Collect and transform data from confluence
        confluence_pages = collect_data_from_confluence(space)
        # Progress indicator
        time.sleep(0.01)
        percent_complete += 5
        progress_text = str(len(confluence_pages)) + ' Seiten geladen.'
        progress_bar.progress(percent_complete, text = progress_text)


        # Split pages
        confluence_pages = split_large_pages(confluence_pages)
        # Progress indicator
        time.sleep(0.01)
        percent_complete += 5
        progress_text = 'aufgeteilt in ' + str(len(confluence_pages)) + ' Seiten.'
        progress_bar.progress(percent_complete, text = progress_text)


        # Turn into data frame for easier processing
        pages_df = pd.DataFrame(confluence_pages, columns = ['space', 'title', 'page_content', 'link', 'num_tokens'])

        # Initialize 'embeddings' as a column of type object, which can hold lists
        pages_df['embeddings'] = None
        pages_df['embeddings'] = pages_df['embeddings'].astype(object)


        # Create embeddings
        i = 0
        total = len(confluence_pages)

        for row in pages_df.itertuples():
            i += 1
            embeddings = create_embeddings(row.page_content, embedding_model)
            pages_df.at[row.Index, 'embeddings'] = embeddings

            # Progress indicator
            percent_complete = 10 + round(i / total * 90)
            progress_text = 'Erzeuge Embeddings (' + str(i) + ' von ' + str(total) + ')'
            progress_bar.progress(percent_complete, text = progress_text)

        # Write csv
        file_name_space = get_file_name_for_space(file_name, space)
        pages_df.to_csv(file_name_space, index = False)

        # Progress indicator
        percent_complete = 100
        progress_text = 'CSV Datei ' + file_name_space + ' geschrieben'
        progress_bar.progress(percent_complete, text = progress_text)

        time.sleep(1)
        progress_bar.empty()

    return pages_df
