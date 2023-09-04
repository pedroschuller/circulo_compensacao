import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import janitor
import os
import re

pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# Limpar dados base
def obter_base(path, ano):
    sheet_nacional = f'AR_{ano}_Distrito_Reg.Autónoma' 
    sheet_int = f'AR_{ano}_Global' 
    # Território nacional
    df_nac = pd.read_excel(path, sheet_name=sheet_nacional, skiprows = 3, nrows = 21) 
    # Europa e fora da europa
    df_int = pd.read_excel(path, sheet_name=sheet_int, skiprows = 3, nrows = 5) 

    # Corrigir nomes
    mapping={"nome do território": "distrito", "distrito/região autónoma": "distrito", 'círculo':'distrito', 'nome do distrito/região autónoma':'distrito'}
    df_nac.rename(columns=mapping, inplace = True)
    df_int.rename(columns=mapping, inplace = True)

    # Filtrar linhas
    df_total = pd.concat([df_nac.loc[(df_nac["código"]!=500000) & (df_nac["código"]!=990000)], 
                        df_int.loc[df_int["código"]==800000],
                        df_int.loc[df_int["código"]==810000], 
                        df_int.loc[df_int["código"]==820000], 
                        df_int.loc[df_int["código"]==900000]])   

    return df_total


# Mandatos e inscritos por circulo
def obter_mandatos(df_total): 
    df_base = pd.concat([df_total.iloc[:,0:2],  df_total.iloc[:,6:9], df_total.iloc[:,13]], axis = 1).reset_index(drop = True)
    df_base["eleitores por mandato"] = df_base["inscritos"]/df_base["mandatos"]

    return df_base


# Resultados dos partidos, nulos e brancos
def obter_votos(df_total): 
    df_partidos = pd.concat([df_total.iloc[:,0:2], df_total.iloc[:,14:], df_total.iloc[:,9:13]], axis = 1).fillna(0)
    df_partidos_treat = df_partidos.pivot_longer(
                            index = ['código','distrito']
                            , names_to = ["partido", "drop1", "drop2"]
                            , values_to = ["votos", "% votos", "mandatos"]
                            , names_pattern = ["^brancos|^nulos|[A-Z]", "% brancos|% nulos|% votantes.*", "mandatos.*"]
                        ).drop(columns=['drop1', 'drop2'])
    
    return df_partidos_treat


# Retirar mandatos aos círculos distritais para o de compensacao
def reduzir(df, tam_circ_comp, min_circ):
    df = df.copy()
    mandatos_atuais = sum(df["mandatos"])

    # Mínimo de deputados por círculo
    df["mandatos"] = df["mandatos"].clip(lower=min_circ) 
    mandatos_corrigidos = sum(df["mandatos"])

    # Se o tamanho mínimo deu deputados, temos que os retirar de outro circulo
    mandatos_a_retirar = mandatos_corrigidos - mandatos_atuais 

    for _ in range(tam_circ_comp + int(mandatos_a_retirar)): 
        # Eleitores por deputado
        df["eleitores_por_mandato"] = df["inscritos"] / df["mandatos"] 
        # Círculos que já não podem perder mais
        df["atingiu_minimo"] = df["mandatos"].apply(lambda x: x == min_circ)
        # Círculo com menos eleitores por deputado
        df.sort_values(["atingiu_minimo", "eleitores_por_mandato"], inplace=True, ignore_index=True) 
        # Retirar mandato
        df.iloc[0, df.columns.get_loc("mandatos")] -= 1  
    
    return df


# Calcular desvio de proporcionalidade entre votos e deputados
def calcular_desvio(df_votos):
    # Agregar votos totais
    df_votos_nacional = df_votos[~df_votos.partido.isin(["brancos", "nulos"])].groupby("partido", as_index =False)[['votos', 'mandatos']].sum()

    # Proporção de votos
    df_votos_nacional["%votos"] = df_votos_nacional["votos"]/sum(df_votos_nacional["votos"]) 
    # Proporção de mandatos 
    df_votos_nacional["%mandatos"] = df_votos_nacional["mandatos"]/sum(df_votos_nacional["mandatos"]) 
    # Desvio absoluto entre os 2
    df_votos_nacional["desvio"] = abs(df_votos_nacional["%votos"]-df_votos_nacional["%mandatos"]) 
    # Desvio total
    soma_dos_desvios = sum(df_votos_nacional["desvio"]) 

    return df_votos_nacional, soma_dos_desvios


def metodo_hondt(df_mandatos, df_votos, circ_comp, incluir_estrangeiros = True):
    df_hondt = df_votos.iloc[:0,:].copy()

    # Retirar nulos e brancos
    df_votos  = df_votos[df_votos['partido'].isin(['nulos', 'brancos']) == False].copy(deep = True) 

    # Inicializar mandatos atribuidos e algoritmo 
    df_votos['mandatos'] = 0 
    df_votos['votos_dhondt'] = df_votos['votos'] 
    
    # Para cada distrito:
    for d in df_mandatos.itertuples(): 
        mandatos_d = d.mandatos
        votos_d = df_votos[df_votos['distrito'] == d.distrito]
        
        mandatos_atribuidos = 0

        # Atribuir mandatos dos círculos distritais
        while mandatos_atribuidos < mandatos_d:
            # Partido a Eleger
            max_v = votos_d['votos_dhondt'].max()
            max_v_index = votos_d[votos_d['votos_dhondt'] == max_v].index[0]

            # Atribuir mandato
            votos_d.at[max_v_index, 'mandatos'] += 1
            mandatos_atribuidos += 1

            # Recalcular d'Hondt
            votos_d.at[max_v_index, 'votos_dhondt'] = votos_d.at[max_v_index, 'votos'] / (votos_d.at[max_v_index, 'mandatos'] + 1)
        
        # Acrescentar resultados do distrito
        df_hondt = pd.concat([df_hondt, votos_d[df_hondt.columns]], ignore_index = True)

    # Agregar todos os votos a nível nacional (com ou sem estrangeiros)
    if incluir_estrangeiros:
        df_compensacao = df_hondt.groupby("partido", as_index=False)[['votos', 'mandatos']].sum()
    else:
        df_compensacao = df_hondt[~df_hondt.distrito.isin(['Europa', 'Fora da Europa'])].groupby("partido", as_index=False)[['votos', 'mandatos']].sum()


    # Inicializar mandatos atribuidos no circulo de compensacao e algoritmo 
    df_compensacao['mandatos_compensacao'] = 0
    df_compensacao['votos_dhondt'] = df_compensacao['votos'] / (df_compensacao['mandatos'] + 1)

    # Atribuir mandatos círculo compensação
    for _ in range(circ_comp):
        # É atribuido o novo mandato ao partido com mais votos a dividir por todos os mandatos já atribuídos
        max_v = df_compensacao['votos_dhondt'].max()
        max_v_index = df_compensacao[df_compensacao['votos_dhondt'] == max_v].index[0]

        # Atribuir mandato
        df_compensacao.at[max_v_index, 'mandatos'] += 1
        df_compensacao.at[max_v_index, 'mandatos_compensacao'] += 1

        # Recalcular d'Hondt
        df_compensacao.at[max_v_index, 'votos_dhondt'] = df_compensacao.at[max_v_index, 'votos'] / (df_compensacao.at[max_v_index, 'mandatos'] + 1)
    
    # Partidos eleitos no círculo de compensação
    eleitos_compensacao = df_compensacao[df_compensacao['mandatos_compensacao'] > 0]['partido'].unique()

    # São dados como perdidos os votos que não elegeram ninguém
    # Se elegeu no círculo de compensação, nenhum voto daquele partido é dado como perdido
    df_perdidos = df_hondt.loc[(df_hondt['mandatos'] == 0) & (~df_hondt['partido'].isin(eleitos_compensacao))].copy(deep = True)
    
    # Adicionar círculo de compensação aos restantes
    df_compensacao['código'] = 999999
    df_compensacao['distrito'] = "Compensação"
    df_compensacao['% votos'] = pd.NA
    df_compensacao = df_compensacao[['código', 'distrito', 'partido', 'votos', '% votos', 'mandatos_compensacao']]
    df_compensacao.rename(columns={"mandatos_compensacao": "mandatos"}, inplace=True)
    df_hondt = pd.concat([df_hondt, df_compensacao], ignore_index=True)
    
    return df_hondt, df_perdidos

# Desenhar gráfico com votos perdidos e desvio de proporcionalidade por tamanho do círculo de compensação
def plot_desvios(df_desvios, eleicao):
    # Criar 2 plots
    fig, (ax0, ax1) = plt.subplots(2,1,figsize = (10, 5)) 

    # Tornar nomes mais legíveis
    df_desvios_nomes = df_desvios.rename(columns={"circulo_compensacao":"Tamanho do Círculo de Compensação"
                                                  , "desvio_proporcionalidade":"Desvio de proporcionalidade (%)"
                                                  , "votos_perdidos": "Votos perdidos"})

    # Dados detalhe
    df_desvios_focus = df_desvios_nomes.loc[df_desvios_nomes['Tamanho do Círculo de Compensação']<=55]

    # Definir valor máximo do eixo dos y
    max_y = max([-(-int(max(df_desvios_focus['Desvio de proporcionalidade (%)'])+1)//5)*5, -(-int(max(df_desvios_focus['Votos perdidos'])+1)//125000)*5])
    max_y2 = max_y * 25000

    # Gráfico detalhe
    df_desvios_focus.plot(x = 'Tamanho do Círculo de Compensação', y = 'Desvio de proporcionalidade (%)', ax = ax0, title = eleicao, ylim = (0, max_y), yticks = range(0,max_y+5,5), grid = True)
    df_desvios_focus.plot(x = 'Tamanho do Círculo de Compensação', y = 'Votos perdidos', ax = ax0, secondary_y = True, mark_right = False, label = 'Votos perdidos (eixo direita)', ylim = (0, max_y2), yticks = range(0,max_y2+125000,125000))
    ax0.set(xlabel=None)
    ax0.get_legend().remove()


    # Gráfico total
    df_desvios_nomes.plot(x = 'Tamanho do Círculo de Compensação', y = 'Desvio de proporcionalidade (%)', ax = ax1, ylim = (0, max_y), yticks = range(0,max_y+5,5)) 
    df_desvios_nomes.plot(x = 'Tamanho do Círculo de Compensação', y = 'Votos perdidos', ax = ax1, secondary_y = True, ylim = (0, max_y2), yticks = range(0,max_y2+125000,125000))  

    # Área zoom
    ax1.fill_between((0,55), 0, max_y, facecolor='grey', alpha=0.1)

    # Linha esquerda
    con1 = ConnectionPatch(xyA=(0, max_y), coordsA=ax1.transData, 
                        xyB=(-2.5, -0.5), coordsB=ax0.transData, color = 'black')
    con1.set_linewidth(0.3)
    fig.add_artist(con1)

    # Linha direita
    con2 = ConnectionPatch(xyA=(55, max_y), coordsA=ax1.transData, 
                        xyB=(57.5, -0.5), coordsB=ax0.transData, color = 'black')
    con2.set_linewidth(0.3)
    fig.add_artist(con2)

    fig.savefig(f'plots\\{eleicao}.jpg')
    return 0


def simular_eleicao(df_mandatos, df_votos, lista_tamanhos_cc, tamanho_circulo_minimo, eleicao, incluir_estrangeiros):

    df_desvios = pd.DataFrame(columns = ['circulo_compensacao', 'desvio_proporcionalidade', 'votos_perdidos'])

    for t in lista_tamanhos_cc:
        df_reduzido = reduzir(df_mandatos, t, tamanho_circulo_minimo)
        df_simulacao, df_perdidos = metodo_hondt(df_reduzido, df_votos, t, incluir_estrangeiros)
        _, desvio = calcular_desvio(df_simulacao)
        votos_perdidos = sum(df_perdidos['votos'])
        df_desvios.loc[len(df_desvios)] = [t, desvio*100.0, votos_perdidos]
        
    # Guardar resultados
    if(len(lista_tamanhos_cc)==1):
        df_simulacao.to_csv(f'simulacoes\\simulacao_{eleicao}_cc_{lista_tamanhos_cc[0]}_mandatos.csv')
    else:
        df_desvios.to_csv(f'simulacoes\\desvios_{eleicao}.csv')
        plot_desvios(df_desvios, eleicao)

    return 0

# Simular todas as eleicoes
def main(eleicoes, tamanho_circulo_minimo, lista_tamanhos_cc = range(0, 231), incluir_estrangeiro = True):
    for ficheiro in eleicoes:
        #ficheiro = eleicoes[0]
        eleicao = os.path.splitext(ficheiro)[0]
        print(eleicao)
        path = f'eleicoes\\{ficheiro}'
        ano = re.search('\d{4}', ficheiro)[0]
        df_total = obter_base(path, ano)
        df_mandatos = obter_mandatos(df_total)
        df_votos = obter_votos(df_total)
        erro = simular_eleicao(df_mandatos, df_votos, lista_tamanhos_cc, tamanho_circulo_minimo, eleicao, incluir_estrangeiro)
        plt.close('all')

    return erro

# Simular sequência de tamanhos do círculo de compensação (min, max+1, [step])
lista_tamanhos_cc = range(0, 231, 1)

## OU ##

# Simular um tamanho
#lista_tamanhos_cc = [40]

# Mínimo de mandatos por círculo distrital
tamanho_circulo_minimo = 2

# Listar eleições a simular
eleicoes = os.listdir('eleicoes')
#eleicoes = [eleicoes[0]]

# Círculos eleitorais do estrangeiro contam para o círculo nacional de compensação?
incluir_estrangeiro = False

if __name__ == "__main__":
   main(eleicoes, tamanho_circulo_minimo, lista_tamanhos_cc, incluir_estrangeiro)