import duckdb
import os
import glob
import json

print("\n🚀 BUSCA GLOBAL POR DADOS...\n")

# 1. Procura TODOS os arquivos .duckdb no projeto
arquivos_db = glob.glob("**/*.duckdb", recursive=True)

for path in arquivos_db:
    try:
        con = duckdb.connect(path, read_only=True)
        # Verifica se a tabela existe nesse arquivo específico
        tabelas = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
        tabelas_nomes = [t[0] for t in tabelas]
        
        print(f"File: {path}")
        print(f" -> Tables found: {tabelas_nomes}")
        
        if 'param_sets' in tabelas_nomes:
            print(f" ✨ ACHEI! A tabela está aqui. Nomes:")
            rows = con.execute("SELECT name FROM param_sets LIMIT 3").fetchall()
            for r in rows: print(f"   - {r[0]}")
        con.close()
    except Exception as e:
        print(f" ❌ Erro ao ler {path}: {e}")
    print("-" * 30)

# 2. Busca o JSON em qualquer lugar
print("\n🔍 BUSCANDO O JSON...")
json_files = glob.glob("**/all_wf_results.json", recursive=True) + glob.glob("../results/**/all_wf_results.json", recursive=True)

if json_files:
    print(f" ✅ Achei o JSON em: {json_files[0]}")
    with open(json_files[0], 'r') as f:
        data = json.load(f)
        primeira_chave = list(data.keys())[0]
        exemplo = data[primeira_chave].get("results", [{}])[0].get("best_param")
        print(f" -> Exemplo no JSON: {exemplo}")
else:
    print(" ❌ JSON não encontrado em nenhuma subpasta.")