from graphviz import Source

# DOT 파일 경로
dot_file_path = 'model_graph'

# DOT 파일을 불러와서 Source 객체로 생성
dot = Source.from_file(dot_file_path)

# DOT 파일을 PDF로 변환 및 저장
dot.render('Phi3_graph', format='pdf')
#dot.render('Phi3_graph', format='svg')

