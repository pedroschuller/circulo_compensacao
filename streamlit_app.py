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

pd.set_option('mode.use_inf_as_na', False)

# Configs streamlit
st.set_page_config(page_title="Círculo de Compensação Nacional")
# Custom CSS rules for better year select display and text links
st.markdown("""
    <style>
        .stSelectbox * { cursor: pointer !important; }
        .stSelectbox div[value] { color: black; }
        div[data-baseweb=popover] * {color: black !important }
        .stMarkdown a { color: #FFF !important;}
        .stMarkdown a:hover { text-decoration: none !important;}
    </style>
""", unsafe_allow_html=True)
st.title("Círculo de Compensação Nacional")
st.header("Simulação da introdução de um mecanismo de compensação nas eleições legislativas")
#st.image('./votos_que_contam.png')
st.write("O sistema eleitoral português enfrenta, há décadas, desafios críticos que necessitam de uma atenção urgente e ponderada. Um dos problemas mais prementes é o desperdício significativo de votos, uma realidade que não só desvirtua a verdadeira vontade expressa nas urnas pelos cidadãos, como também alimenta o fenómeno preocupante do voto útil ou táctico. Este cenário, onde centenas de milhares de votos não contribuem para a eleição de qualquer deputado, mina a confiança no sistema democrático, favorecendo desproporcionalmente os partidos maiores, em detrimento de uma representação parlamentar verdadeiramente plural e reflectiva da diversidade política do eleitorado.")

st.write("A Iniciativa Liberal, consciente desta problemática, propôs uma solução ambiciosa e equitativa: a introdução de um círculo nacional de compensação. Esta proposta visa garantir que cada voto conta, independentemente do distrito a que pertence, promovendo um sistema eleitoral mais justo, proporcional e alinhado com o espírito da Constituição da República Portuguesa. Este mecanismo permitiria uma distribuição de mandatos mais fiel às preferências dos eleitores, incentivando a participação cívica e fortalecendo a legitimidade dos representantes eleitos.")

st.write("Para ilustrar de forma clara e transparente o impacto desta proposta, esta ferramenta permite visualizar, de maneira intuitiva e fundamentada, como a introdução de um círculo de compensação nacional poderia ter alterado os resultados das eleições legislativas desde 2005. Convido-te a compreender a profundidade da questão e a juntares-te na defesa de um sistema eleitoral que verdadeiramente honre os princípios democráticos de representatividade e proporcionalidade.")

#Substituir coligações, fusões ou rebrandings pelo maior partido (simplificação)
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

# Partidos da esquerda para a direita (discutível mas suficiente)
ordem_partidos = ['MAS', 'B.E.', 'MRPP', 'POUS', 'PCP-PEV', 'PTP', #esquerda
                  'L', 'PS', 'JPP', 'PAN', 'PURP', 'VP',  'R.I.R.', #centro-esquerda
                  'P.H.', 'MPT', 'NC', 'MMS', 'MEP', 'PDA', 'PDR', #centro
                  'IL', 'PPD/PSD', 'AD', 'A', 'CDS-PP', 'PPM', #centro-direita
                  'PND', 'CH', 'ADN', 'PNR'] #direita

# Abreviar distritos para o plot
mapping_distritos = {'Castelo Branco':'C. Branco',
                     'Viana do Castelo': 'V. Castelo',
                     'Fora da Europa':'F. Europa',
                     'Compensação':'Comp.'}

# Cores aproximadas dos partidos em RGBA
cores = ['black', 'black', 'darkred', 'darkred', 'red', 'darkred', 
         'lightgreen', 'pink', 'lightgreen', 'green', 'orange', 'purple',  'green', 
         'orange', 'green', 'yellow', 'darkblue', 'green', 'blue', 'black', 
         'cyan', 'orange', 'cyan', 'orange', 'blue', 'darkblue', 
         'red', 'darkblue', 'yellow', 'red']
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
def plot_comparacao(df_votos, df_simulacao, df_perdidos, df_mandatos, df_reduzido, eleicao, tamanho_cc):

    # Preparar dados
    df_merge_votos = pd.merge(df_votos.dropna(), right = df_simulacao, on = ['código', 'distrito', 'partido'], how = 'right', suffixes = ('', '_cc'))
    df_merge_votos['votos_perdidos'] = np.where(df_merge_votos.mandatos == 0, df_merge_votos.votos, 0)
    df_merge_votos = pd.merge(df_merge_votos, right = df_perdidos, on = ['código', 'distrito', 'partido'], how = 'left', suffixes = ('', '_perdidos_cc')).fillna(0)
    df_distritos = df_merge_votos.groupby(['distrito'])[['votos', 'mandatos', 'mandatos_cc', 'votos_perdidos', 'votos_perdidos_cc']].agg('sum')
    df_merge_votos = df_merge_votos.groupby(['partido'])[['votos', 'mandatos', 'mandatos_cc', 'votos_perdidos', 'votos_perdidos_cc']].agg('sum')
    df_merge_votos['%votos_nao_convertidos'] = 100.0 * df_merge_votos.votos_perdidos / df_merge_votos.votos
    df_merge_votos['%votos_nao_convertidos_cc'] = 100.0 * df_merge_votos.votos_perdidos_cc / df_merge_votos.votos
    df_merge_votos['votos_por_deputado'] = df_merge_votos.votos / df_merge_votos.mandatos 
    df_merge_votos['votos_por_deputado_cc'] = df_merge_votos.votos / df_merge_votos.mandatos_cc

    df_reorganizacao_mandatos = pd.merge(df_mandatos[['distrito', 'mandatos']], right = df_reduzido[['distrito', 'mandatos']], on = 'distrito', suffixes=('', '_cc'))
    df_reorganizacao_mandatos['diferenca'] = df_reorganizacao_mandatos.mandatos - df_reorganizacao_mandatos.mandatos_cc
    df_reorganizacao_mandatos.loc[len(df_reorganizacao_mandatos)] = ['Compensação', 0, 0, tamanho_cc]


    # Hemiciclo
    st.write("A proporcionalidade entre mandatos e votos é um pilar fundamental para a integridade e legitimidade de qualquer sistema democrático representativo. Este princípio assegura que a distribuição de mandatos parlamentares reflete de maneira justa e precisa a vontade expressa pelos eleitores nas urnas, garantindo que cada voto contribui efetivamente para a configuração do órgão legislativo. Quando há uma desconexão entre a proporção de votos recebidos pelos partidos e o número de mandatos que lhes é atribuído, emerge um desequilíbrio que pode distorcer a representação política, marginalizar vozes minoritárias e, consequentemente, enfraquecer a confiança dos cidadãos no sistema eleitoral e nas instituições democráticas. Portanto, a busca pela proporcionalidade entre mandatos e votos é não apenas uma questão de justiça eleitoral, mas também um imperativo para a consolidação e fortalecimento da democracia.")
    cores_usar = df_cores[df_cores.index.isin(df_merge_votos.index)]['cor']

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    help = plt.imread('mandatos_votos.png')
    imagebox = OffsetImage(help, zoom = 0.14)
    ab = AnnotationBbox(imagebox, (1.25, 1.03), frameon = False)
    axs[0].add_artist(ab)
    
    plot_hemiciclo(axs[0], df_merge_votos['mandatos'][cores_usar.index], df_merge_votos['votos'][cores_usar.index], cores_usar.values, 'Atual', cores_usar.index)
    plot_hemiciclo(axs[1], df_merge_votos['mandatos_cc'][cores_usar.index], df_merge_votos['votos'][cores_usar.index], cores_usar.values, 'Com Círculo de Compensação', cores_usar.index)

    fig.suptitle("Como ficaria o parlamento?")
    plt.show()
    st.pyplot(fig, bbox_inches = transforms.Bbox([[0.7, 2.5], [11.5, 6.2],]))

    # Votos para eleger um deputado por partido
    st.write("A disparidade no número de votos necessários para eleger um deputado entre diferentes partidos é uma questão preocupante que evidencia distorções no sistema eleitoral, comprometendo a equidade e a justiça representativa. Esta incongruência resulta, muitas vezes, numa representação parlamentar que não espelha fielmente a vontade do eleitorado. Partidos com menos expressão, apesar de receberem um apoio significativo nas urnas, encontram-se sub-representados, necessitando de um número desproporcionalmente elevado de votos para garantir um mandato. Por outro lado, partidos maiores beneficiam deste sistema, obtendo mais mandatos do que a proporção de votos recebidos justificaria.")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    df_merge_votos.sort_values(['votos_por_deputado', 'votos'], ascending = [False, True], inplace = True)
    hbar_colors = cores_usar.reindex(df_merge_votos.index.values)

    # Define the columns and titles to iterate over
    titles = ['Atual', 'Círculo de Compensação']
    columns = ['votos_por_deputado', 'votos_por_deputado_cc']

    with pd.option_context('mode.use_inf_as_na', True):
        xlim = np.ceil(df_merge_votos[['votos_por_deputado', 'votos_por_deputado_cc']].dropna().max().max()/10000)*10000

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


    # Reorganização de mandatos por distrito
    st.write("Com a introdução de um círculo nacional de compensação, a distribuição de mandatos entre os distritos eleitorais sofre algumas alterações para assegurar a proporcionalidade. Neste modelo, os mandatos destinados ao círculo de compensação são extraídos incrementalmente dos distritos eleitorais existentes. O processo inicia-se pela transferência de um mandato do distrito que apresenta o menor rácio de votos por deputado eleito, promovendo assim uma redistribuição ponderada. Este procedimento é repetido até que o número total de mandatos no círculo de compensação seja atingido. Importante salientar que, durante este processo de realocação, cada distrito eleitoral mantém, no mínimo, dois mandatos, garantindo que todas as regiões continuem a ter uma representação parlamentar básica. Esta abordagem visa não apenas preservar a representatividade de cada distrito, mas também criar um mecanismo de compensação que reflete de forma mais precisa e justa a vontade expressa pelos eleitores em todo o território nacional.")
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
    fig.suptitle("Como ficariam os círculos eleitorais?")
    st.pyplot(fig)

    

    # Votos perdidos (em percentagem) por distrito
    st.write("Os distritos mais pequenos do interior, ilhas e estrangeiro enfrentam uma taxa de desperdício de votos inaceitável, uma realidade que compromete gravemente a equidade e a integridade do processo democrático nestas regiões. Devido à sua menor densidade populacional e ao consequente número reduzido de mandatos atribuídos, muitos votos nestes distritos não resultam na eleição de representantes, sendo, na prática, votos inutilizados. Este cenário não só desfavorece os eleitores destas áreas, que veem a sua voz diluída no conjunto nacional, como também alimenta um ciclo de desencanto e apatia política, dado que os cidadãos podem sentir que seu voto tem um impacto limitado ou nulo.")
    df_distritos = df_distritos[~df_distritos.index.isin(['Compensação'])]
    df_distritos.rename(index = mapping_distritos, inplace = True)
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

    fig.suptitle("Que percentagem de votos não serve para eleger ninguém, por distrito?")
    st.pyplot(fig)


    # Votos perdidos por distrito
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
    fig.suptitle("Quantos votos não servem para eleger ninguém, por distrito?")
    st.pyplot(fig)


    # Votos que não serviram para eleger por partido
    st.write("Os partidos maiores em Portugal beneficiam significativamente do atual sistema eleitoral, sendo capazes de converter uma maior percentagem de votos a eles atribuídos em representantes eleitos. Este fenómeno contrasta acentuadamente com a realidade enfrentada pelos partidos menores, nos quais um número considerável de votos acaba por não se traduzir em mandatos, resultando em uma representação parlamentar que não espelha de forma justa o apoio popular que recebem. Esta discrepância amplifica o poder dos partidos dominantes, enquanto marginaliza e limita a influência de vozes políticas emergentes ou minoritárias, comprometendo a pluralidade e a diversidade essenciais para uma democracia robusta e vibrante.")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    df_merge_votos.sort_values(['%votos_nao_convertidos', 'votos'], ascending=[False, True], inplace=True)

    # Define the columns and titles to iterate over
    columns = ['%votos_nao_convertidos', '%votos_nao_convertidos_cc']

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
    st.write("A proposta de introdução de um círculo nacional de compensação visa, fundamentalmente, reduzir o número total de votos perdidos que o atual sistema eleitoral inevitavelmente produz. Enquanto no modelo vigente, centenas de milhares de votos não têm impacto na composição final da Assembleia da República, a implementação do círculo de compensação nacional reduz este valor em uma ordem de grandeza. Este ajuste não só valoriza o princípio democrático fundamental de que cada voto tem peso, e um peso igual, mas também revitaliza a confiança dos eleitores no processo eleitoral, incentivando uma participação mais ativa e consciente nas urnas,  contribuíndo para o voto convicto e não apenas útil, refletindo de forma mais fidedigna a diversidade e pluralidade da vontade popular no panorama político nacional.")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the bars and add data labels
    for i, label in enumerate(['votos_perdidos', 'votos_perdidos_cc']):
        axs[i].bar(eleicao, sum(df_merge_votos[label]), color='red' if label == 'votos_perdidos' else 'blue')
        axs[i].set_xlabel('Ano')
        axs[i].set_title('Atual' if label == 'votos_perdidos' else 'Se houvesse Círculo de Compensação')

        # Add data labels
        axs[i].text(eleicao, sum(df_merge_votos[label]), format_k(sum(df_merge_votos[label])), ha='center', va='bottom', fontsize=8)

    ylim = np.ceil(sum(df_merge_votos['votos_perdidos'])/100000)*100000
    plt.setp(axs[0], ylim=(0,ylim))
    plt.setp(axs[1], ylim=(0,ylim))
    fig.suptitle('Quantos votos se perdem, no total?')
    st.pyplot(fig)


# Simular resultados de uma eleição dada uma lista de tamanhos de círculo de compensação
def simular_eleicao(df_mandatos, df_votos, tamanho_cc, tamanho_circulo_minimo, eleicao, incluir_estrangeiros):

    df_desvios = pd.DataFrame(columns = ['circulo_compensacao', 'desvio_proporcionalidade', 'votos_perdidos'])

    df_reduzido = reduzir(df_mandatos, tamanho_cc, tamanho_circulo_minimo)
    df_simulacao, df_perdidos = metodo_hondt(df_reduzido, df_votos, tamanho_cc, incluir_estrangeiros)
    
    plot_comparacao(df_votos, df_simulacao, df_perdidos, df_mandatos, df_reduzido, eleicao, tamanho_cc)


    return df_perdidos


# Convert numbers to 'k' format
def format_k(x):
    return f"{x/1000:.0f}k" if x >= 1000 else str(x)


# Simular 
def main(eleicao, tamanho_circulo_minimo, tamanho_cc = range(0, 231), incluir_estrangeiros = True):

    df_mandatos = pd.read_csv(f'./eleicoes/mandatos/{eleicao}.csv')
    df_votos = pd.read_csv(f'./eleicoes/votos/{eleicao}.csv')

    df_perdidos  = simular_eleicao(df_mandatos, df_votos, tamanho_cc, tamanho_circulo_minimo, eleicao, incluir_estrangeiros)

    url = 'https://www.parlamento.pt/ActividadeParlamentar/Paginas/DetalheIniciativa.aspx?BID=243359'
    st.write("Não se pode continuar a ignorar o elefante na sala do nosso sistema eleitoral. É crucial agir para fortalecer a nossa democracia, garantindo que cada voto conta. Convido todos os cidadãos conscientes e comprometidos com um sistema eleitoral mais representativo e justo a visitar esta [proposta](%s) detalhada no site do Parlamento Português. Não há portugueses de segunda, não pode haver votos de segunda." % url)
    #st.image('./votos_que_contam.png')
    st.divider()
    st.write('\u00a9 Pedro Schuller 2023')  


# Listar eleições a simular
eleicao = st.selectbox(
    'Que eleição deseja simular?',
    ('2024', '2022', '2019', '2015', '2011', '2009', '2005'))

# Mínimo de mandatos por círculo distrital
tamanho_circulo_minimo = 2

# Círculos eleitorais do estrangeiro contam para o círculo nacional de compensação?
incluir_estrangeiros = st.toggle('Votos nos círculos eleitorais internacionais contam para o círculo nacional de compensação?', value = True)

# simulação não pode retirar mais deputados do que o mínimo 
tamanho_maximo_circulo_compensacao = 230 - (20 + 2 * incluir_estrangeiros) * tamanho_circulo_minimo - 4 * operator.not_(incluir_estrangeiros)

# Simular um tamanho
tamanho_cc = st.slider('Número de deputados no círculo de compensação nacional', 0, tamanho_maximo_circulo_compensacao, 40)

if __name__ == "__main__":
   main(eleicao, tamanho_circulo_minimo, tamanho_cc, incluir_estrangeiros)

   
