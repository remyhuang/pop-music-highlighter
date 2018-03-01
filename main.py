from model import MusicHighlighter

def extract():
    model = MusicHighlighter()
    model.extract(length=30, save_score=True, save_thumbnail=True, save_wav=True)

if __name__ == '__main__':
    extract()