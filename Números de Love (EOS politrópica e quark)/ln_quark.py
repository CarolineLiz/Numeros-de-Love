import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from numba import njit
import time


# Declaração de constantes
MeV_fm3_to_SI = 10**6 * constants.e * (10**(-15))**(-3)
pressao_SI_to_GU = constants.c**(-4) * constants.G
densidade_energia_SI_to_GU = constants.c**(-4) * constants.G
massa_solar = 1.48e3
n_estrelas = 50
l = 2
precisao = 1e-20

# Constantes da EOS de quarks
B = 130 * MeV_fm3_to_SI * pressao_SI_to_GU
a2 = (100 * MeV_fm3_to_SI * densidade_energia_SI_to_GU)**(1 / 2)
a4 = 0.6


# Definição da EOS e sua derivada
@njit
def p_eos(rho):
    p_eos = (
        (1 / 3) * (rho - 4 * B) - (a2**2 / (12 * np.pi**2 * a4)) * (
            1 + (1 + ((16 * np.pi**2 * a4) / a2**2) * (rho - B))**(1 / 2)
        )
    )
    return p_eos

@njit
def rho_eos(p):
    rho_eos = (
        3 * p + 4 * B + ((3 * a2**2) / (4 * np.pi**2 * a4)) * (
            1 + (1 + ((16 * np.pi**2 * a4) / (3 * a2**2)) * (p + B))**(1 / 2)
        )
    )
    return rho_eos

@njit
def dp_drho(rho):
    dp_drho = (
        (1 / 3) - (2 / 3) * (
            (1 + ((16 * np.pi**2 * a4) / a2**2) * (rho - B))**(-1 / 2)
        )
    )
    return dp_drho

# Definição das equações diferenciais para densidade, massa e para a pertubação
@njit
def densidade(r, m, rho):
    return -((rho + p_eos(rho)) / dp_drho(rho)) * ((m + 4 * np.pi * r**3 * p_eos(rho)) / (r**2 - 2 * m * r))

@njit
def massa(r, rho):
    return 4 * np.pi * r**2 * rho

@njit
def fpertub(r, m, rho, z, pertub):
    return (-z * (2 / r + (2 * m / r**2 + 4 * np.pi * r * (p_eos(rho) - rho)) / (1 - 2 * m / r)) - pertub * (-(l * (l + 1)) / (r**2 * (1 - 2 * m / r)) + (4 * np.pi / (1 - 2 * m / r))
            * (5 * rho + 9 * p_eos(rho) + (rho + p_eos(rho)) / dp_drho(rho)) - (2 * (m + 4 * np.pi * r**3 * p_eos(rho)) / (r**2 - 2 * m * r))**2))

# Função principal acelerada com o numba
@njit
def main():

    # Declaração das listas
    lrho = []
    lm = []
    lr = []
    lhp = []
    lpertub = []
    lz = []
    lk2 = []
    lc = []
    lm_estrela = []
    lr_estrela = []

    # Cálculo da faixa de densidade central utilizada
    rho_center_max = 2.376364e-9        # Densidade central máxima [m^-2]
    p_center_max = p_eos(rho_center_max)
    p_center_space = p_center_max * np.logspace(-4.0, 1.0, n_estrelas)
    rho_center_space = rho_eos(p_center_space)

    # Calcula para diferentes valores de densidade central
    for rho in rho_center_space:

        # Limpa as listas utilizadas nos cálculos
        lrho.clear()
        lm.clear()
        lr.clear()
        lhp.clear()
        lpertub.clear()
        lz.clear()

        # Constantes iniciais
        h = 1.0
        r = 1e-16
        m = 0.0
        pertub = r**l
        z = l * (r**(l - 1))
        y = 0.0
        c = 0.0

        # Salvando os primeiros valores nas listas
        lrho.append(rho)
        lm.append(m)
        lr.append(r)
        lhp.append(h)
        lpertub.append(pertub)
        lz.append(z)

        # Condições para determinação do raio da estrela
        while p_eos(lrho[-1]) > precisao:
            while p_eos(lrho[-1]) > 0:

                # Método de Runge-Kutta para as três equações simultaneamente
                k0m = h * massa(lr[-1], lrho[-1])
                k0dens = h * densidade(lr[-1], lm[-1], lrho[-1])
                k0z = h * fpertub(lr[-1], lm[-1], lrho[-1], lz[-1], lpertub[-1])
                k0pertub = h * lz[-1]

                k1m = h * massa(lr[-1] + h / 2, lrho[-1] + k0dens / 2)
                k1dens = h * densidade(lr[-1] + h / 2, lm[-1] + k0m / 2, lrho[-1] + k0dens / 2)
                k1z = h * fpertub(lr[-1] + h / 2, lm[-1] + k0m / 2, lrho[-1] + k0dens / 2, lz[-1] + k0z / 2, lpertub[-1] + k0pertub / 2)
                k1pertub = h * (lz[-1] + k0z / 2)

                k2m = h * massa(lr[-1] + h / 2, lrho[-1] + k1dens / 2)
                k2dens = h * densidade(lr[-1] + h / 2, lm[-1] + k1m / 2, lrho[-1] + k1dens / 2)
                k2z = h * fpertub(lr[-1] + h / 2, lm[-1] + k1m / 2, lrho[-1] + k1dens / 2, lz[-1] + k1z / 2, lpertub[-1] + k1pertub / 2)
                k2pertub = h * (lz[-1] + k1z / 2)

                k3m = h * massa(lr[-1] + h, lrho[-1] + k2dens)
                k3dens = h * densidade(lr[-1] + h, lm[-1] + k2m, lrho[-1] + k2dens)
                k3z = h * fpertub(lr[-1] + h, lm[-1] + k2m, lrho[-1] + k2dens, lz[-1] + k2z, lpertub[-1] + k2pertub)
                k3pertub = h * (lz[-1] + k2z)

                # Calcula os valores deste passo de cálculo
                m = lm[-1] + (1 / 6 * (k0m + 2 * k1m + 2 * k2m + k3m))
                rho = lrho[-1] + (1 / 6 * (k0dens + 2 * k1dens + 2 * k2dens + k3dens))
                z = lz[-1] + 1 / 6 * (k0z + 2 * k1z + 2 * k2z + k3z)
                pertub = lpertub[-1] + 1 / 6 * (k0pertub + 2 * k1pertub + 2 * k2pertub + k3pertub)
                r = r + h

                # Adiciona os valores nas listas
                lrho.append(rho)
                lm.append(m)
                lr.append(r)
                lhp.append(h)
                lpertub.append(pertub)
                lz.append(z)

            # O valor da pressão ficou negativo, logo, excluı́mos todos os últimos valores calculados nesse raio especı́fico, e diminuı́mos o h pela metade
            lrho.pop()
            lm.pop()
            lr.pop()
            lhp.pop()
            lpertub.pop()
            lz.pop()
            h = h / 2

        # Calcula y, c, m_estrela, r_estrela
        y = lr[-1] * lz[-1] / lpertub[-1]
        c = lm[-1] / lr[-1]
        m_estrela = lm[-1]
        r_estrela = lr[-1]

        # Adiciona c, m_estrela e r_estrela em listas
        lc.append(c)
        lm_estrela.append(m_estrela)
        lr_estrela.append(r_estrela)

        # Utilização dos valores encontrados de y e c para o cálculo do número de Love k2
        if c >= 0.0026:
            k2 = (8 * c**5 / 5) * (1 - 2 * c)**2 * (2 + 2 * c * (y - 1) - y) / (2 * c * (6 - 3 * y + 3 * c * (5 * y - 8)) + 4 * c**3 *
                                                                                (13 - 11 * y + c * (3 * y - 2) + 2 * c**2 * (1 + y)) + 3 * (1 - 2 * c)**2 * (2 - y + 2 * c * (y - 1)) * np.log(1 - 2 * c))
            lk2.append(k2)
        # Expansão em Taylor para compacidade pequena
        else:
            k2 = (2 - y) / (2 * (3 + y)) + (5 * (-6 + 2 * y + y**2) * c) / (2 * (3 + y)**2)
            lk2.append(k2)

    return lr_estrela, lm_estrela, lc, lk2, p_center_space, rho_center_space


# Função para plotar os gráficos
def plot_graph(lr_estrela, lm_estrela, lc, lk2, p_center_space, rho_center_space):

    # Converte listas para arrays numpy
    array_r_estrela = np.array(lr_estrela)
    array_m_estrela = np.array(lm_estrela)
    array_c = np.array(lc)
    array_k2 = np.array(lk2)
    array_p_center = np.array(p_center_space)
    array_rho_center = np.array(rho_center_space)

    # Faz o plot da curva M vs R
    plt.figure()
    plt.plot(array_r_estrela / 10**3, array_m_estrela / massa_solar, linewidth=1, label="Curva calculada", marker='.')
    plt.title("Curva Massa-Raio")
    plt.xlabel("R [km]")
    plt.ylabel("$M [M_{\\odot}]$")
    plt.legend()
    plt.show()

    # Faz o plot da curva k2 vs C
    plt.figure()
    plt.plot(array_c, array_k2, linewidth=1, label="Curva calculada", marker='.')
    plt.title("Curva k2 vs C")
    plt.xlabel("C = M/R [adimensional]")
    plt.ylabel("k2 [adimensional]")
    plt.legend()
    plt.show()

    # Faz o plot da curva M vs rho_central
    plt.figure()
    plt.plot(array_rho_center, array_m_estrela / massa_solar, linewidth=1, label="Curva calculada", marker='.')
    plt.title("Curva Massa vs Densidade central")
    plt.xlabel("Densidade central $[m^{-2}]$")
    plt.ylabel("$M [M_{\\odot}]$")
    plt.legend()
    plt.show()

    # Faz o plot da curva R vs rho_central
    plt.figure()
    plt.plot(array_rho_center, array_r_estrela / 10**3, linewidth=1, label="Curva calculada", marker='.')
    plt.title("Curva Raio vs Densidade central")
    plt.xlabel("Densidade central $[m^{-2}]$")
    plt.ylabel("R [km]")
    plt.legend()
    plt.show()

    # Faz o plot da curva C vs rho_central
    plt.figure()
    plt.plot(array_rho_center, array_c, linewidth=1, label="Curva calculada", marker='.')
    plt.title("Curva Compacidade vs Densidade central")
    plt.xlabel("Densidade central $[m^{-2}]$")
    plt.ylabel("C = M/R [adimensional]")
    plt.legend()
    plt.show()

    # Faz o plot da curva k2 vs rho_central
    plt.figure()
    plt.plot(array_rho_center, array_k2, linewidth=1, label="Curva calculada", marker='.')
    plt.title("Curva k2 vs Densidade central")
    plt.xlabel("Densidade central $[m^{-2}]$")
    plt.ylabel("k2 [adimensional]")
    plt.legend()
    plt.show()


# Executa a função principal e gera os gráficos
if __name__ == "__main__":
    tempo_inicio = time.time()
    lr_estrela, lm_estrela, lc, lk2, p_center_space, rho_center_space = main()
    delta_tempo = time.time() - tempo_inicio
    print(f"Tempo de execução = {delta_tempo} s")
    plot_graph(lr_estrela, lm_estrela, lc, lk2, p_center_space, rho_center_space)
