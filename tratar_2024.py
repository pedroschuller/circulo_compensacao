import pandas as pd
import janitor

mapping_partidos = {'E':'PNR',
                    'P.N.R.':'PNR',
                    'MPT-P.H.':'MPT',
                    'PPV':'CH',
                    'PPV/CDC':'CH',
                    'CDS-PP.PPM':'CDS-PP',
                    'L/TDA':'L',
                    'PPD/PSD.CDS-PP':'AD',
                    'PTP-MAS':'PTP',
                    'PPD/PSD.CDS-PP.PPM':'AD',
                    'PCTP/MRPP':'MRPP'}


def unique_non_null(s):
    return s.dropna().unique()


df_total = pd.read_csv('resultados_eleicoes_AR_2024_Ultimo.csv')
mapping={"nome do território": "distrito", "distrito/região autónoma": "distrito", 'círculo':'distrito', 'nome do distrito/região autónoma':'distrito'}
df_total.rename(columns=mapping, inplace = True)
df_total = df_total.loc[df_total["código"].str.isnumeric()]
df_total["código"] = df_total["código"].astype(int)
df_total = df_total.loc[df_total["código"] % 10000 == 0]
df_total = df_total.loc[(df_total["código"]!=500000) & (df_total["código"]!=600000) & (df_total["código"]!=990000)]

# Identifying and renaming the columns
for i in range(1, 21):  # Adjust based on your actual data structure
    party_col = f'opção {i}'
    df_total.rename(columns={
        f'votos {i}': f'{unique_non_null(df_total[party_col])[0]}',
        #f'Party{i}_Percentage': f'{df[party_col].iloc[0]}_Percentage',
        #f'Party{i}_Mandates': f'{df[party_col].iloc[0]}_Mandates'
    }, inplace=True)
    df_total.drop(columns=[party_col], inplace=True)

df_total = pd.concat([df_total.iloc[:,0:2], df_total.iloc[:,21:], df_total.iloc[:,15:19]], axis = 1).fillna(0)

df_votos = df_total.pivot_longer(
                        index = ['código','distrito']
                        , names_to = ["partido", "drop1", "drop2"]
                        , values_to = ["votos", "% votos", "mandatos"]
                        , names_pattern = ["^brancos|^nulos|[A-Z]", "% brancos|% nulos|% votos*", "mandatos *"]
                    ).drop(columns=['drop1', 'drop2'])


df_votos['partido'].replace(mapping_partidos, inplace = True)
df_votos = df_votos.groupby(['código', 'distrito', 'partido']).sum().reset_index()

df_votos.to_csv('eleicoes/votos/2024.csv')