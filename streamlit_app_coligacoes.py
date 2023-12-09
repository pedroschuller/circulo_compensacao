# Imports
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
import streamlit as st


# Configs
pd.set_option('mode.use_inf_as_na', False)


# Configs streamlit
st.set_page_config(page_title="Coligação Legislativas")
st.markdown("""
    <style>
        .stSelectbox * { cursor: pointer !important; }
        .stSelectbox div[value] { color: black; }
        div[data-baseweb=popover] * {color: black !important }
        .stMarkdown a { color: #FFF !important;}
        .stMarkdown a:hover { text-decoration: none !important;}
        .stMultiselect * { cursor: pointer !important; }
        .stMultiselect div[value] { color: black; }
    </style>
""", unsafe_allow_html=True)


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


# Resultados a nível nacional
def obter_nacional(df_votos):
    df_nacional = df_votos[['partido', 'votos', 'mandatos']].copy(deep = True)
    df_nacional.partido = np.where(df_nacional.partido.isin(grandes_partidos), df_nacional.partido ,'O/B/N')
    df_nacional = df_nacional.groupby('partido').sum()
    df_nacional['flag_obn'] = (df_nacional.index == 'O/B/N')
    df_nacional.sort_values(['flag_obn', 'votos'], ascending = [True, False], inplace = True)
    df_nacional.reset_index(inplace = True)
    df_nacional['% votos'] = 100.0 * df_nacional.votos / np.sum(df_nacional.votos)
    df_nacional['% votos arred.'] = np.round(df_nacional['% votos'], 1)
    return df_nacional


# Ajustar votos com base na sondagem
def ajustar_votacao(df_votos, df_sondagem, df_nacional):
    df_nacional = pd.merge(df_nacional, right = df_sondagem, how = 'left', on = 'partido')
    df_nacional['multiplicador'] = df_nacional['votação'] / df_nacional['% votos arred.']

    df_votos_ajust = pd.merge(df_votos, right = df_nacional[['partido', 'multiplicador']], on = 'partido')
    df_votos_ajust['votos_old'] = df_votos_ajust['votos']
    df_votos_ajust['votos'] = df_votos_ajust['votos'] * df_votos_ajust['multiplicador']
    return df_votos_ajust


# Algoritmo Método d'Hondt
def metodo_hondt(df_mandatos, df_votos, df_coligacao):
    df_hondt = df_votos.iloc[:0,:].copy()

    # Retirar nulos e brancos
    df_coligacao = df_coligacao[df_coligacao['partido'].isin(['nulos', 'brancos']) == False].copy(deep = True) 

    # Inicializar mandatos atribuidos e algoritmo 
    df_coligacao['mandatos'] = 0 
    df_coligacao['votos_dhondt'] = df_coligacao['votos'] 
    
    # Para cada distrito:
    for d in df_mandatos.itertuples(): 
        mandatos_d = d.mandatos
        votos_d = df_coligacao[df_coligacao['distrito'] == d.distrito]
        
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
    
    return df_hondt


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


# Desenhar gráficos de comparação entre a situação atual e uma coligação
def plot_comparacao(df_votos, df_simulacao, eleicao, coligacao):

    df_cores = pd.DataFrame(cores, ordem_partidos, columns = ['cor'])


    # Preparar dados
    df_merge_votos = pd.merge(df_votos.dropna(), right = df_simulacao, on = ['código', 'distrito', 'partido'], how = 'outer', suffixes = ('', '_col'))
    df_merge_votos = df_merge_votos[~df_merge_votos['partido'].isin(['brancos', 'nulos'])]
    df_merge_votos['votos_perdidos'] = np.where(df_merge_votos.mandatos == 0, df_merge_votos.votos, 0)
    df_merge_votos['votos_perdidos_col'] = np.where(df_merge_votos.mandatos_col == 0, df_merge_votos.votos_col, 0)
    #df_merge_votos = pd.merge(df_merge_votos, right = df_perdidos, on = ['código', 'distrito', 'partido'], how = 'left', suffixes = ('', '_perdidos_col')).fillna(0)
    df_distritos = df_merge_votos.groupby(['distrito'])[['votos', 'votos_col', 'mandatos', 'mandatos_col', 'votos_perdidos', 'votos_perdidos_col']].agg('sum')
    df_merge_votos = df_merge_votos.groupby(['partido'])[['votos', 'votos_col', 'mandatos', 'mandatos_col', 'votos_perdidos', 'votos_perdidos_col']].agg('sum')
    df_merge_votos['%votos_nao_convertidos'] = 100.0 * df_merge_votos.votos_perdidos / df_merge_votos.votos
    df_merge_votos['%votos_nao_convertidos_col'] = 100.0 * df_merge_votos.votos_perdidos_col / df_merge_votos.votos_col
    df_merge_votos['votos_por_deputado'] = df_merge_votos.votos / df_merge_votos.mandatos 
    df_merge_votos['votos_por_deputado_col'] = df_merge_votos.votos_col / df_merge_votos.mandatos_col

    df_merge_votos.loc[df_merge_votos['votos_por_deputado'].isna(), 'votos_por_deputado'] = np.inf
    df_merge_votos.loc[df_merge_votos['votos_por_deputado_col'].isna(), 'votos_por_deputado_col'] = np.inf


    # Hemiciclo
    df_cores = pd.concat([df_cores[0:int(len(df_cores)*5/8)], pd.DataFrame(['teal'], ['/'.join(coligacao)], columns = ['cor']), df_cores[int(len(df_cores)*5/8):len(df_cores)]])
    cores_usar = df_cores[df_cores.index.isin(df_merge_votos.index)]['cor']


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    help = plt.imread('mandatos_votos.png')
    imagebox = OffsetImage(help, zoom = 0.14)
    ab = AnnotationBbox(imagebox, (1.25, 1.03), frameon = False)
    axs[0].add_artist(ab)
    
    plot_hemiciclo(axs[0], df_merge_votos['mandatos'][cores_usar.index], df_merge_votos['votos'][cores_usar.index], cores_usar.values, 'Atual', cores_usar.index)
    plot_hemiciclo(axs[1], df_merge_votos['mandatos_col'][cores_usar.index], df_merge_votos['votos_col'][cores_usar.index], cores_usar.values, 'Com Coligação', cores_usar.index)

    fig.suptitle("Como ficaria o parlamento?")
    plt.show()
    st.pyplot(fig, bbox_inches = transforms.Bbox([[0.7, 2.5], [11.5, 6.2],]))

    # Votos para eleger um deputado por partido

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    df_merge_votos.sort_values(['votos_por_deputado', 'votos_por_deputado_col', 'votos'], ascending = [False, False, True], inplace = True)
    hbar_colors = cores_usar.reindex(df_merge_votos.index.values)

    # Define the columns and titles to iterate over
    titles = ['Atual', 'Coligação']
    columns = ['votos_por_deputado', 'votos_por_deputado_col']

    with pd.option_context('mode.use_inf_as_na', True):
        xlim = np.ceil(df_merge_votos[['votos_por_deputado', 'votos_por_deputado_col']].dropna().max().max()/10000)*10000

    # Iterate over the columns and titles to create the subplots
    for i, (col, title) in enumerate(zip(columns, titles)):
        bars = axs[i].barh(df_merge_votos.index, df_merge_votos[col], color=hbar_colors)
        axs[i].set_xlabel('Votos necessários para eleger um deputado')
        axs[i].set_title(title)

        # Add labels with thousands separator to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.13 * xlim if width <= 0.5 * xlim else width - 0.165 * xlim  # Adjust the offset as needed
            label_color = 'black' if width <= 0.5 * xlim else 'white'
            axs[i].text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:,.0f}', 
                        color=label_color, va='center', ha='right' if width <= 0.5 * xlim else 'left')


    plt.setp(axs[0], xlim=(0,xlim))
    plt.setp(axs[1], xlim=(0,xlim))
    fig.suptitle("Quantos votos seriam necessários para eleger um deputado?")
    st.pyplot(fig)
    

    # Votos perdidos (em percentagem) por distrito
    df_distritos = df_distritos[~df_distritos.index.isin(['Compensação'])]
    df_distritos.rename(index = mapping_distritos, inplace = True)
    df_distritos['%votos_perdidos'] = 100.0*df_distritos['votos_perdidos']/df_distritos['votos']
    df_distritos['%votos_perdidos_col'] = 100.0*df_distritos['votos_perdidos_col']/df_distritos['votos']
    df_distritos.sort_values(['%votos_perdidos'], inplace=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Define the columns and titles to iterate over
    columns = ['%votos_perdidos', '%votos_perdidos_col']

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

    fig.suptitle("Que percentagem de votos não serve para eleger ninguém, por distrito?")
    st.pyplot(fig)


    # Votos perdidos por distrito
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Define the columns and titles to iterate over
    columns = ['votos_perdidos', 'votos_perdidos_col']

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
    fig.suptitle("Quantos votos não servem para eleger ninguém, por distrito?")
    st.pyplot(fig)


    # Votos que não serviram para eleger por partido
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    df_merge_votos.sort_values(['%votos_nao_convertidos', '%votos_nao_convertidos_col', 'votos'], ascending=[False, False, True], inplace=True)
    hbar_colors = cores_usar.reindex(df_merge_votos.index.values)


    # Define the columns and titles to iterate over
    columns = ['%votos_nao_convertidos', '%votos_nao_convertidos_col']

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

    fig.suptitle("Que percentagem de votos, por partido, não servem para nada?")
    st.pyplot(fig)

     
    # Total de votos perdidos

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the bars and add data labels
    for i, label in enumerate(['votos_perdidos', 'votos_perdidos_col']):
        axs[i].bar(eleicao, sum(df_merge_votos[label]), color='red' if label == 'votos_perdidos' else 'blue')
        axs[i].set_xlabel('Ano')
        axs[i].set_title('Atual' if label == 'votos_perdidos' else 'Se houvesse Coligação')

        # Add data labels
        axs[i].text(eleicao, sum(df_merge_votos[label]), format_k(sum(df_merge_votos[label])), ha='center', va='bottom', fontsize=8)

    ylim = np.ceil(sum(df_merge_votos['votos_perdidos'])/100000)*100000
    plt.setp(axs[0], ylim=(0,ylim))
    plt.setp(axs[1], ylim=(0,ylim))
    fig.suptitle('Quantos votos se perdem, no total?')
    st.pyplot(fig)


# Simular resultados de uma eleição dadas as sondagens e uma coligação
def simular_eleicao(df_mandatos, df_votos, df_votos_ajust, df_coligacao, eleicao, coligacao):

    df_votos_ajust = metodo_hondt(df_mandatos, df_votos, df_votos_ajust)    

    df_coligacao = metodo_hondt(df_mandatos, df_votos, df_coligacao)    
    
    plot_comparacao(df_votos_ajust, df_coligacao, eleicao, coligacao)

    pass


# Convert numbers to 'k' format
def format_k(x):
    return f"{x/1000:.0f}k" if x >= 1000 else str(x)


# Coligar partidos
def coligar(df_votos, coligacao, distritos_coligacao):
    df_coligacao = df_votos.copy()
    df_coligacao['coligados'] = df_coligacao['partido'].isin(coligacao) & df_coligacao['distrito'].isin(distritos_coligacao)
    df_coligacao.loc[df_coligacao['coligados'],'partido'] = '/'.join(coligacao)
    df_coligacao = df_coligacao.groupby(['código', 'distrito', 'partido']).sum()[['votos', '% votos']].reset_index()
    return df_coligacao


# Simular 
def main():
    st.title("Simulador de coligações eleitorais")

    # Listar eleições a simular
    eleicao = st.selectbox(
        'Eleição a simular:',
        ('2022', '2019', '2015', '2011', '2009', '2005'))
        
    df_mandatos = pd.read_csv(f'./eleicoes/mandatos/{eleicao}.csv')
    df_votos = pd.read_csv(f'./eleicoes/votos/{eleicao}.csv')

    partidos = df_votos.partido.unique()
    partidos = partidos[(partidos != 'brancos') & (partidos != 'nulos')]
    
    df_nacional = obter_nacional(df_votos)

    df_a_editar = df_nacional[['partido', '% votos arred.']]
    df_a_editar.columns = ['partido', 'votação']

    st.write("Inserir sondagem:")
    df_sondagem = st.data_editor(df_a_editar, disabled = 'partido', hide_index = True)

    checksum = sum(df_sondagem['votação'])

    if((checksum > 101.0) | (checksum < 99.0)):
        st.write(f'Os votos devem totalizar 100%! De momento somam {checksum}')
    else:

        df_votos_ajust = ajustar_votacao(df_votos, df_sondagem, df_nacional)

        # Simular uma coligação
        coligacao = st.multiselect(
            'Que coligação deseja simular?',
            (partidos), ['PPD/PSD', 'IL', 'CDS-PP'])

        distritos_coligacao = st.multiselect(
            'Em que distritos há coligação',
            ['Aveiro', 'Beja', 'Braga', 'Bragança', 'Castelo Branco', 'Coimbra',
            'Évora', 'Faro', 'Guarda', 'Leiria', 'Lisboa', 'Portalegre',
            'Porto', 'Santarém', 'Setúbal', 'Viana do Castelo', 'Vila Real',
            'Viseu', 'Madeira', 'Açores', 'Europa', 'Fora da Europa'],
            ['Beja', 'Bragança', 'Castelo Branco', 'Coimbra',
            'Évora', 'Guarda', 'Portalegre',
            'Santarém', 'Viana do Castelo', 'Vila Real',
            'Viseu', 'Madeira', 'Açores', 'Europa', 'Fora da Europa'])

        df_coligacao = coligar(df_votos_ajust, coligacao, distritos_coligacao)
        simular_eleicao(df_mandatos, df_votos, df_votos_ajust, df_coligacao, eleicao, coligacao)

    st.divider()
    st.write('\u00a9 Pedro Schuller 2023')  


# Substituir coligações, fusões ou rebrandings pelo maior partido (simplificação)
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


# Abreviar distritos para o plot
mapping_distritos = {'Castelo Branco':'C. Branco',
                     'Viana do Castelo': 'V. Castelo',
                     'Fora da Europa':'F. Europa',
                     'Compensação':'Comp.'}


# Partidos da esquerda para a direita (discutível mas suficiente)
ordem_partidos = ['MAS', 'B.E.', 'MRPP', 'POUS', 'PCP-PEV', 'PTP', #esquerda
                'L', 'PS', 'JPP', 'PAN', 'PURP', 'VP',  'R.I.R.', #centro-esquerda
                'P.H.', 'MPT', 'NC', 'MMS', 'MEP', 'PDA', 'PDR', #centro
                'IL', 'PPD/PSD', 'A', 'CDS-PP', 'PPM', #centro-direita
                'PND', 'CH', 'ADN', 'PNR'] #direita


# Cores aproximadas dos partidos em RGBA
cores = ['black', 'black', 'darkred', 'darkred', 'red', 'darkred', 
        'lightgreen', 'pink', 'lightgreen', 'green', 'orange', 'purple',  'green', 
        'orange', 'green', 'yellow', 'darkblue', 'green', 'blue', 'black', 
        'cyan', 'orange', 'cyan', 'blue', 'darkblue', 
        'red', 'darkblue', 'yellow', 'red']


# Partidos a excluir de "outros"
grandes_partidos = ['B.E.', 'PCP-PEV', 
                    'L', 'PS', 'PAN',  
                    'IL', 'PPD/PSD', 'CDS-PP',
                    'CH'] 


if __name__ == "__main__":
   main()

