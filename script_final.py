import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import transforms
import janitor
import operator
import os
import re
import numpy as np


#Substituir coligações, fusões ou rebrandings pelo maior partido (simplificação)
mapping_partidos = {'E':'PNR',
                    'P.N.R.':'PNR',
                    'MPT-P.H.':'MPT',
                    'PPV':'CH',
                    'PPV/CDC':'CH',
                    'CDS-PP.PPM':'CDS-PP',
                    'L/TDA':'L',
                    'PPD/PSD.CDS-PP':'PPD/PSD',
                    'PTP-MAS':'PTP',
                    'PPD/PSD.CDS-PP.PPM':'PPD/PSD',
                    'PCTP/MRPP':'MRPP'}

# Partidos da esquerda para a direita (discutível mas suficiente)
ordem_partidos = ['MAS', 'B.E.', 'MRPP', 'POUS', 'PCP-PEV', 'PTP', 'L', 'PS', 'JPP', 'PAN', 'PURP', 'VP',  'R.I.R.', 'P.H.', 'MPT', 'NC', 'MMS', 'MEP', 'PDA', 'PDR', 'PPD/PSD', 'IL', 'A', 'CDS-PP', 'PPM', 'PND', 'CH', 'ADN', 'PNR']

# Abreviar distritos para o plot
mapping_distritos = {'Castelo Branco':'C. Branco',
                     'Viana do Castelo': 'V. Castelo',
                     'Fora da Europa':'F. Europa',
                     'Compensação':'Comp.'}

# Cores aproximadas dos partidos em RGBA
cores = ['black', 'black', 'darkred', 'darkred', 'red', 'darkred', 'lightgreen', 'pink', 'lightgreen', 'green', 'orange', 'purple',  'green', 'orange', 'green', 'yellow', 'darkblue', 'green', 'blue', 'black', 'orange', 'cyan', 'cyan', 'blue', 'darkblue', 'red', 'darkblue', 'yellow', 'red']
df_cores = pd.DataFrame(cores, ordem_partidos, columns = ['cor'])


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

    # Filtrar linhas #PROBLEM
    df_total = pd.concat([df_nac.loc[(df_nac["código"]!=500000) & (df_nac["código"]!=990000)], 
                        df_int.loc[df_int["código"]==800000],
                        df_int.loc[df_int["código"]==810000], 
                        df_int.loc[df_int["código"]==820000], 
                        df_int.loc[df_int["código"]==900000]])   

    return df_total


# Limpar dados base
def obter_base_concelho(path, ano):

    sheet_concelho = f'AR_{ano}_Concelho' 

    df_concelho = pd.read_excel(path, sheet_name=sheet_concelho, skiprows = 3, nrows = 308) 

    # Corrigir nomes
    df_concelho.rename(columns={"nome do território":"concelho"}, inplace= True)
    df_concelho = df_concelho.iloc[1:]

    return df_concelho


# Mandatos e inscritos por circulo
def obter_mandatos(df_total): 
    df_base = pd.concat([df_total.iloc[:,0:2],  df_total.iloc[:,6:9], df_total.iloc[:,13]], axis = 1).reset_index(drop = True)
    df_base["eleitores por mandato"] = df_base["inscritos"]/df_base["mandatos"]

    return df_base


# Resultados dos partidos, nulos e brancos
def obter_votos(df_total): 
    df_votos = pd.concat([df_total.iloc[:,0:2], df_total.iloc[:,14:], df_total.iloc[:,9:13]], axis = 1).fillna(0)
    df_votos = df_votos.pivot_longer(
                            index = ['código','distrito']
                            , names_to = ["partido", "drop1", "drop2"]
                            , values_to = ["votos", "% votos", "mandatos"]
                            , names_pattern = ["^brancos|^nulos|[A-Z]", "% brancos|% nulos|% votantes.*", "mandatos.*"]
                        ).drop(columns=['drop1', 'drop2'])
    
    df_votos['partido'].replace(mapping_partidos, inplace = True)
    df_votos = df_votos.groupby(['código', 'distrito', 'partido']).sum().reset_index()
    
    return df_votos

# Resultados dos partidos, nulos e brancos
def obter_votos_concelho(df_concelho): 
    df_votos_concelho = pd.concat([df_concelho.iloc[:,0:2], df_concelho.iloc[:,13:], df_concelho.iloc[:,9:13]], axis = 1).fillna(0)
    df_votos_concelho = df_votos_concelho.pivot_longer(
                            index = ['código','concelho']
                            , names_to = ["partido", "drop1"]
                            , values_to = ["votos", "% votos"]
                            , names_pattern = ["^brancos|^nulos|[A-Z]", "% brancos|% nulos|% votantes.*"]
                        ).drop(columns=['drop1'])
    
    df_votos_concelho['partido'].replace(mapping_partidos, inplace = True)
    df_votos_concelho = df_votos_concelho.groupby(['código', 'concelho', 'partido']).sum().reset_index()
    df_votos_concelho['codigo_distrito'] = np.where(df_votos_concelho['código']<200000, np.floor(df_votos_concelho['código']/10000).astype(int)*10000, np.floor(df_votos_concelho['código']/100000).astype(int)*100000)
    
    return df_votos_concelho


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


# Algoritmo Método d'Hondt
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

    # Gráfico total
    df_desvios_nomes.plot(x = 'Tamanho do Círculo de Compensação', y = 'Desvio de proporcionalidade (%)', ax = ax0, mark_right = False,  title = eleicao, ylim = (0, max_y), yticks = range(0,max_y+5,5), ) 
    df_desvios_nomes.plot(x = 'Tamanho do Círculo de Compensação', y = 'Votos perdidos', ax = ax0, secondary_y = True, mark_right = False, label = 'Votos perdidos (eixo direita)', ylim = (0, max_y2), yticks = range(0,max_y2+125000,125000))  
    ax0.set(xlabel=None)

    # Gráfico detalhe
    df_desvios_focus.plot(x = 'Tamanho do Círculo de Compensação', y = 'Desvio de proporcionalidade (%)', ax = ax1, ylim = (0, max_y), yticks = range(0,max_y+5,5), grid = True)
    df_desvios_focus.plot(x = 'Tamanho do Círculo de Compensação', y = 'Votos perdidos', ax = ax1, secondary_y = True, ylim = (0, max_y2), yticks = range(0,max_y2+125000,125000))
    ax1.get_legend().remove()

    # Área zoom
    ax0.fill_between((0,55), 0, max_y, facecolor='grey', alpha=0.1)

    # Linha esquerda
    con1 = ConnectionPatch(xyA=(0, 0), coordsA=ax0.transData, 
                        xyB=(-2.5, max_y), coordsB=ax1.transData, color = 'black')
    con1.set_linewidth(0.3)
    fig.add_artist(con1)

    # Linha direita
    con2 = ConnectionPatch(xyA=(55, 0), coordsA=ax0.transData, 
                        xyB=(57.5, max_y), coordsB=ax1.transData, color = 'black')
    con2.set_linewidth(0.3)
    fig.add_artist(con2)

    # Adicionar legendas dos eixos y
    fig.text(x=0.075,y=0.275,s="Desvio de proporcionalidade (%)", rotation = 'vertical')
    fig.text(x=0.975,y=0.38,s="Votos perdidos", rotation = 'vertical')

    fig.savefig(f'plots\\{eleicao}.jpg')
    return 0


# Função gráfico hemiciclo 
def plot_hemiciclo(ax, mandatos, votos, cores, title, ordem_partidos):
    mandatos = np.append(mandatos, np.sum(mandatos))
    votos = np.append(votos, np.sum(votos))
    cores = np.append(cores, 'white')  # Add a white color for the gap
    ordem_partidos = np.append(ordem_partidos, '')  # Add an empty label for the gap

    # Create the pie charts with edges and labels
    wedges1, _ = ax.pie(mandatos, colors=cores, startangle=180, radius=1.0, counterclock=False, 
           wedgeprops=dict(width=0.3, edgecolor='k'))
    wedges2, _ =ax.pie(votos, colors=cores, startangle=180, radius=0.7, counterclock=False, 
           wedgeprops=dict(width=0.3, edgecolor='k'), labels=None)
    
    # Set the aspect ratio and title
    ax.set(aspect="equal", title=title)

    # Remove edge for the white wedge
    for wedge in wedges1 + wedges2:
        if wedge.get_facecolor() == (1.0, 1.0, 1.0, 1.0):  # Check if the wedge color is white
            wedge.set_edgecolor("white")  # Set edge color to white

    # Adding labels for mandatos > 0
    cumulative_sum = (np.cumsum(mandatos) - mandatos/2)*180/230  # Calculate position of text label
    last_y = 0
    for i, (wedge, label) in enumerate(zip(wedges1, ordem_partidos)):
        if mandatos[i] > 0 and cores[i] != 'white':  # Only add label if mandatos > 0 and color is not white
            x = 1.2 * np.cos(np.radians(180 - cumulative_sum[i]))  # x position
            y = 1.2 * np.sin(np.radians(180 - cumulative_sum[i]))  # y position
            
            # Adjust y position to prevent overlap
            if i > 1 and abs(last_y - y) < 0.05:  # Adjust the threshold as needed
                y = last_y - 0.07  # Adjust the offset as needed
            
            ax.text(x, y, f"{label}: {int(mandatos[i])}", ha='center', va='center', fontsize=8)  # Adjust fontsize as needed
            last_y = y  # Store the last y position


# Desenhar gráficos de comparação entre a situação atual e a introdução de um círculo de compensação
def plot_comparacao(df_votos, df_simulacao, df_perdidos, df_mandatos, df_reduzido, eleicao, lista_tamanhos_cc, df_votos_concelho):

    # Preparar dados
    df_merge_votos = pd.merge(df_votos.dropna(), right = df_simulacao, on = ['código', 'distrito', 'partido'], how = 'right', suffixes = ('', '_cc'))
    df_merge_votos['votos_perdidos'] = np.where(df_merge_votos.mandatos == 0, df_merge_votos.votos, 0)
    df_merge_votos = pd.merge(df_merge_votos, right = df_perdidos, on = ['código', 'distrito', 'partido'], how = 'left', suffixes = ('', '_perdidos_cc')).fillna(0)
    df_distritos = df_merge_votos.groupby(['distrito'])[['votos', 'mandatos', 'mandatos_cc', 'votos_perdidos', 'votos_perdidos_cc']].agg('sum')
        
    df_votos_concelho = df_votos_concelho.loc[~df_votos_concelho['partido'].isin(['nulos', 'brancos'])]
    df_perdidos_concelho = df_merge_votos.merge(df_votos_concelho, left_on=['código', 'partido'], right_on = ['codigo_distrito','partido'], suffixes=('', '_concelho'))

    df_merge_votos = df_merge_votos.groupby(['partido'])[['votos', 'mandatos', 'mandatos_cc', 'votos_perdidos', 'votos_perdidos_cc']].agg('sum')
    df_merge_votos['%votos_nao_convertidos'] = 100.0 * df_merge_votos.votos_perdidos / df_merge_votos.votos
    df_merge_votos['%votos_nao_convertidos_cc'] = 100.0 * df_merge_votos.votos_perdidos_cc / df_merge_votos.votos
    df_merge_votos['votos_por_deputado'] = df_merge_votos.votos / df_merge_votos.mandatos 
    df_merge_votos['votos_por_deputado_cc'] = df_merge_votos.votos / df_merge_votos.mandatos_cc

    df_reorganizacao_mandatos = pd.merge(df_mandatos[['distrito', 'mandatos']], right = df_reduzido[['distrito', 'mandatos']], on = 'distrito', suffixes=('', '_cc'))
    df_reorganizacao_mandatos['diferenca'] = df_reorganizacao_mandatos.mandatos - df_reorganizacao_mandatos.mandatos_cc
    df_reorganizacao_mandatos.loc[len(df_reorganizacao_mandatos)] = ['Compensação', 0, 0, lista_tamanhos_cc[0]]



    # Hemiciclo
    cores_usar = df_cores[df_cores.index.isin(df_merge_votos.index)]['cor']
    plt.close('all') 
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_hemiciclo(axs[0], df_merge_votos['mandatos'][cores_usar.index], df_merge_votos['votos'][cores_usar.index], cores_usar.values, 'Atual', cores_usar.index)
    plot_hemiciclo(axs[1], df_merge_votos['mandatos_cc'][cores_usar.index], df_merge_votos['votos'][cores_usar.index], cores_usar.values, 'Com Círculo de Compensação', cores_usar.index)
    fig.suptitle(eleicao)
    help = plt.imread('mandatos_votos.png')
    imagebox = OffsetImage(help, zoom = 0.14)
    ab = AnnotationBbox(imagebox, (1.25, 1.03), frameon = False)
    axs[0].add_artist(ab)
    fig.savefig(f'plots\\proporcionalidade_{eleicao}_cc_{lista_tamanhos_cc[0]}.jpg', bbox_inches = transforms.Bbox([[0.7, 2.5], [11.5, 6.2],]))


    # Votos que não serviram para eleger por partido

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    df_merge_votos.sort_values(['%votos_nao_convertidos', 'votos'], ascending=[False, True], inplace=True)
    hbar_colors = cores_usar.reindex(df_merge_votos.index.values)

    # Define the columns and titles to iterate over
    columns = ['%votos_nao_convertidos', '%votos_nao_convertidos_cc']
    titles = ['Atual', 'Círculo de Compensação']

    # Iterate over the columns and titles to create the subplots
    for i, (col, title) in enumerate(zip(columns, titles)):
        bars = axs[i].barh(df_merge_votos.index, df_merge_votos[col], color=hbar_colors)
        axs[i].set_xlabel('Votos não convertidos em cada partido (%)')
        axs[i].set_title(title)

        # Add percentage labels to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 15 if width <= 75 else width - 15 # Adjust the offset as needed
            label_color = 'black' if width <= 75 else 'white'
            axs[i].text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                        color=label_color, va='center', ha='right' if width <= 75 else 'left')


    fig.suptitle(eleicao)
    fig.savefig(f'plots\\votos_nao_convertidos_{eleicao}_cc_{lista_tamanhos_cc[0]}.jpg')


    # Votos para eleger um deputado por partido

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    df_merge_votos.sort_values(['votos_por_deputado', 'votos'], ascending = [False, True], inplace = True)

    # Define the columns and titles to iterate over
    columns = ['votos_por_deputado', 'votos_por_deputado_cc']

    # Iterate over the columns and titles to create the subplots
    for i, (col, title) in enumerate(zip(columns, titles)):
        bars = axs[i].barh(df_merge_votos.index, df_merge_votos[col], color=hbar_colors)
        axs[i].set_xlabel('Votos necessários para eleger um deputado')
        axs[i].set_title(title)

        # Add labels with thousands separator to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.13 * axs[i].get_xlim()[1] if width <= 0.5 * axs[i].get_xlim()[1] else width - 0.165 * axs[i].get_xlim()[1]  # Adjust the offset as needed
            label_color = 'black' if width <= 0.5 * axs[i].get_xlim()[1] else 'white'
            axs[i].text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:,.0f}', 
                        color=label_color, va='center', ha='right' if width <= 0.5 * axs[i].get_xlim()[1] else 'left')


    fig.suptitle(eleicao)
    fig.savefig(f'plots\\votos_por_deputado_{eleicao}_cc_{lista_tamanhos_cc[0]}.jpg')


    # Reorganização de mandatos por distrito

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    df_reorganizacao_mandatos.sort_values(['mandatos', 'mandatos_cc'], inplace = True)
    df_reorganizacao_mandatos['distrito'].replace(mapping_distritos, inplace = True)
    hbar=axs[0].barh(df_reorganizacao_mandatos.distrito, df_reorganizacao_mandatos.mandatos)
    axs[0].bar_label(hbar)
    axs[0].set_xlabel('Mandatos por distrito')
    axs[0].set_title('Atual')
    axs[1].barh(df_reorganizacao_mandatos.distrito, df_reorganizacao_mandatos.mandatos_cc)
    cores_mandatos = ['red']*(len(df_reorganizacao_mandatos)-1)
    cores_mandatos.insert(0, 'green')
    # Create the bar chart
    axs[1].barh(df_reorganizacao_mandatos.distrito, df_reorganizacao_mandatos.diferenca, left=df_reorganizacao_mandatos.mandatos_cc, color=cores_mandatos)

    # Iterate over the bars, and add labels
    for index, value in enumerate(df_reorganizacao_mandatos.distrito):
        # Get width of the 'mandatos_cc' bar
        width_cc = df_reorganizacao_mandatos.mandatos_cc.iloc[index]
        # Get width of the 'diferenca' bar
        width_diff = df_reorganizacao_mandatos.diferenca.iloc[index]
        
        # Position label for 'mandatos_cc' closer to the end of the bar segment
        label_x_pos_cc = width_cc - (width_cc * 0.1)  # Adjust the multiplier as needed to position the label
        
        # Add text label inside 'mandatos_cc' bar segment
        axs[1].text(label_x_pos_cc, index, str(width_cc), color='white', ha='right', va='center')
        
        # Check if 'diferenca' is not zero before adding label
        if width_diff != 0:
            # Position label for 'diferenca' outside the bar, offset by the width of 'mandatos_cc'
            label_x_pos_diff = width_cc + width_diff + (abs(width_diff) * 0.1)  # Adjust the multiplier as needed to position the label
            
            # Invert the sign of 'diferenca' and set color to red (or green for the last element)
            if index == 0:
                label_diff = '+' + str(abs(width_diff))
                color_diff = 'green'
            else:
                label_diff = '-' + str(abs(width_diff))
                color_diff = 'red'
            
            # Add text label for 'diferenca' outside the bar segment
            axs[1].text(label_x_pos_diff, index, label_diff, color=color_diff, ha='left', va='center')

    axs[1].set_xlabel('Mandatos por distrito')
    axs[1].set_title('Círculo de Compensação')
    fig.suptitle(eleicao)
    fig.savefig(f'plots\\reorg_mandatos_{eleicao}_cc_{lista_tamanhos_cc[0]}.jpg')

    
    # Votos perdidos por distrito
    df_distritos = df_distritos[~df_distritos.index.isin(['Compensação'])]
    df_distritos.rename(index = mapping_distritos, inplace = True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Define the columns and titles to iterate over
    columns = ['votos_perdidos', 'votos_perdidos_cc']

    # Iterate over the columns and titles to create the subplots
    for i, (col, title) in enumerate(zip(columns, titles)):
        bars = axs[i].barh(df_distritos.index, df_distritos[col])
        axs[i].set_xlabel('Votos perdidos por distrito')
        axs[i].set_title(title)
            
        # Add labels with 'k' format to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 5000   # Adjust the offset as needed
            label_color = 'black'
            axs[i].text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width/1000:.0f}k', 
                        color=label_color, va='center', ha='right')


    xlim = np.ceil(np.max(df_distritos['votos_perdidos'])/10000)*10000
    plt.setp(axs[0], xlim=(0,xlim))
    plt.setp(axs[1], xlim=(0,xlim))
    fig.suptitle(eleicao)
    fig.savefig(f'plots\\votos_perdidos_distrito_{eleicao}_cc_{lista_tamanhos_cc[0]}.jpg')


    # Votos perdidos (em percentagem) por distrito
    df_distritos['%votos_perdidos'] = 100.0*df_distritos['votos_perdidos']/df_distritos['votos']
    df_distritos['%votos_perdidos_cc'] = 100.0*df_distritos['votos_perdidos_cc']/df_distritos['votos']
    df_distritos.sort_values(['%votos_perdidos'], inplace=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Define the columns and titles to iterate over
    columns = ['%votos_perdidos', '%votos_perdidos_cc']

    # Iterate over the columns and titles to create the subplots
    for i, (col, title) in enumerate(zip(columns, titles)):
        bars = axs[i].barh(df_distritos.index, df_distritos[col])
        axs[i].set_xlabel('Votos perdidos por distrito (%)')
        axs[i].set_title(title)
            
        # Add labels with 'k' format to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 15 if width <= 75 else width - 15 # Adjust the offset as needed
            label_color = 'black' if width <= 75 else 'white'
            axs[i].text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                        color=label_color, va='center', ha='right' if width <= 75 else 'left')
    
    plt.setp(axs[0], xlim=(0,100))
    plt.setp(axs[1], xlim=(0,100))

    fig.suptitle(eleicao)
    fig.savefig(f'plots\\votos_perdidos_perc_distrito_{eleicao}_cc_{lista_tamanhos_cc[0]}.jpg')

    df_merge_votos.to_csv(f'simulacoes\\votos_perdidos_partido_{eleicao}_cc_{lista_tamanhos_cc[0]}_mandatos.csv')
    df_reorganizacao_mandatos.to_csv(f'simulacoes\\reorganizacao_mandatos_{eleicao}_cc_{lista_tamanhos_cc[0]}_mandatos.csv')
    df_distritos.to_csv(f'simulacoes\\votos_perdidos_distrito_{eleicao}_cc_{lista_tamanhos_cc[0]}_mandatos.csv')


    # análise ao concelho (votos perdidos)
    df_perdidos_concelho['votos_perdidos_concelho'] = np.where(df_perdidos_concelho['votos_perdidos']>0, df_perdidos_concelho['votos_concelho'], 0)
    df_perdidos_concelho['votos_perdidos_concelho_cc'] = np.where(df_perdidos_concelho['votos_perdidos_cc']>0, df_perdidos_concelho['votos_concelho'], 0)
    df_perdidos_concelho = df_perdidos_concelho.groupby(['código','distrito','código_concelho','concelho'])[['votos_concelho','votos_perdidos_concelho','votos_perdidos_concelho_cc']].sum().reset_index()

    df_perdidos_concelho['%_votos_perdidos_concelho'] = df_perdidos_concelho['votos_perdidos_concelho']/df_perdidos_concelho['votos_concelho']
    df_perdidos_concelho['%_votos_perdidos_concelho_cc'] = df_perdidos_concelho['votos_perdidos_concelho_cc']/df_perdidos_concelho['votos_concelho']
    df_perdidos_concelho.to_csv(f'simulacoes\\votos_perdidos_concelho_{eleicao}_cc_{lista_tamanhos_cc[0]}_mandatos.csv')

    return 0


# Simular resultados de uma eleição dada uma lista de tamanhos de círculo de compensação
def simular_eleicao(df_mandatos, df_votos, lista_tamanhos_cc, tamanho_circulo_minimo, eleicao, incluir_estrangeiros, df_votos_concelho):

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
        plot_comparacao(df_votos, df_simulacao, df_perdidos, df_mandatos, df_reduzido, eleicao, lista_tamanhos_cc, df_votos_concelho)
    else:
        df_desvios.to_csv(f'simulacoes\\desvios_{eleicao}.csv')
        plot_desvios(df_desvios, eleicao)

    return df_perdidos


# Convert numbers to 'k' format
def format_k(x):
    return f"{x/1000:.0f}k" if x >= 1000 else str(x)


# Simular todas as eleicoes
def main(eleicoes, tamanho_circulo_minimo, lista_tamanhos_cc = range(0, 231), incluir_estrangeiros = True):
    total_perdidos = pd.DataFrame(columns = ['ano', 'votos_perdidos', 'votos_perdidos_cc'])
    for ficheiro in eleicoes:
        #ficheiro = eleicoes[0]
        eleicao = os.path.splitext(ficheiro)[0]
        print(eleicao)
        path = f'eleicoes\\{ficheiro}'
        ano = re.search('\d{4}', ficheiro)[0]
        df_total = obter_base(path, ano)
        df_mandatos = obter_mandatos(df_total)
        df_votos = obter_votos(df_total)
        df_concelho = obter_base_concelho(path, ano)
        df_votos_concelho = obter_votos_concelho(df_concelho)

        df_perdidos  = simular_eleicao(df_mandatos, df_votos, lista_tamanhos_cc, tamanho_circulo_minimo, eleicao, incluir_estrangeiros, df_votos_concelho)


        sum(df_votos.loc[(df_votos['mandatos'] == 0)]['votos'])
        total_perdidos.loc[len(total_perdidos)] = [ano
                                                   , sum(df_votos.loc[(df_votos['mandatos'] == 0)&(~df_votos['partido'].isin(['brancos','nulos']))]['votos'])
                                                   , sum(df_perdidos['votos'])]        
        plt.close('all')

    if(len(lista_tamanhos_cc)==1):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # Plot the bars and add data labels
        for i, label in enumerate(['votos_perdidos', 'votos_perdidos_cc']):
            axs[i].bar(total_perdidos['ano'], total_perdidos[label], color='red' if label == 'votos_perdidos' else 'blue')
            axs[i].set_xlabel('Ano')
            axs[i].set_title('Atual' if label == 'votos_perdidos' else 'Se houvesse Círculo de Compensação')

            # Add data labels
            for x, y in zip(total_perdidos['ano'], total_perdidos[label]):
                axs[i].text(x, y, format_k(y), ha='center', va='bottom', fontsize=8)

        ylim = np.ceil(np.max(total_perdidos['votos_perdidos'])/100000)*100000
        plt.setp(axs[0], ylim=(0,ylim))
        plt.setp(axs[1], ylim=(0,ylim))
        fig.suptitle('Votos perdidos ao longo dos anos')
        fig.savefig(f'plots\\votos_perdidos_por_ano_{lista_tamanhos_cc[0]}.jpg')
        plt.close('all')

        total_perdidos.to_csv(f'simulacoes\\votos_perdidos_por_ano_{lista_tamanhos_cc[0]}.csv')




    return 0


# Mínimo de mandatos por círculo distrital
tamanho_circulo_minimo = 2

# Círculos eleitorais do estrangeiro contam para o círculo nacional de compensação?
incluir_estrangeiros = True

# simulação não pode retirar mais deputados do que o mínimo 
tamanho_maximo_circulo_compensacao = 230 - (20 + 2 * incluir_estrangeiros) * tamanho_circulo_minimo - 4 * operator.not_(incluir_estrangeiros)

# Simular sequência de tamanhos do círculo de compensação (min, max+1, [step])
#lista_tamanhos_cc = range(0, tamanho_maximo_circulo_compensacao+1, 1)

## OU ##

# Simular um tamanho
lista_tamanhos_cc = [40]

# Listar eleições a simular
eleicoes = os.listdir('eleicoes')
eleicoes.pop(5)
#eleicoes = [eleicoes[5]]

if __name__ == "__main__":
   main(eleicoes, tamanho_circulo_minimo, lista_tamanhos_cc, incluir_estrangeiros)