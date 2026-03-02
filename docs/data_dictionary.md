# Dicionário de dados (Gold)

## Chaves
- sector: código setorial (NACE)
- year: ano de referência (int)

## Variável-alvo (exemplo)
- alue_added_clv05_meur: valor agregado (proxy de desempenho econômico setorial)

## Indicadores cloud
- cloud_intensity: indicador anual (macro) para DE
- cloud_intensity_sector: indicador variável por setor (quando disponível)

## Observação importante
- É comum existirem NaN nos indicadores cloud após filtros e junções (cobertura incompleta).  
- Na fase de ML, esses valores são tratados por imputação para evitar falhas de treino.
