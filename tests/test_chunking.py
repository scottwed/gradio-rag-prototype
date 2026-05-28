from inspect import currentframe, getframeinfo
from pathlib import Path
from sys import stderr

from loguru import logger

from vector_db.db_mgmt import chunk_text

logger.remove()  # Remove all existing handlers
logger.add(stderr, level="INFO")  # Prevent debug and lower from appearing on console
logger.add("proto_rag.log", rotation="5 MB", retention=10)  # Write detailed logs with rotation

case_folder = Path(getframeinfo(currentframe()).filename).parent

def test_markdown_header_1():
    file_path = case_folder.joinpath('simple1.md')
    text_input = file_path.read_text(encoding="utf-8")
    result = chunk_text(text=text_input, doc_type=file_path.suffix)
    expected = [
        '# One | ## Empty | ### MostlyEmpty\n1-1 text\n1-2 text',
        '# One | ## Two\n2-1 text\n2-2 text',
        '# One | ## Two | ### Three\n3-1 text\n3-2.text',
        '# One | ## Two | ### Three.2\n3.2-1 text\n3.2-2 text',
        '# One | ## Two.2\n2.2-1 text\n2.2-2 text',
    ]
    assert result == expected

if __name__ == '__main__':
    test_markdown_header_1()