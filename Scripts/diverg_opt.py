# coding: utf-8

"""
This script has been adapted from :
Segmentation Forward/Backward; IRIT- 2013 - Maxime Le Coz
by Timothy Pommée and Régine Andre-Obrect (IRIT - 2021)
"""

from collections import deque
from sys import argv, exit
from getopt import GetoptError, getopt
import time

import numpy as np
from numpy import spacing
from scipy.io.wavfile import read as wavread

try:
    from numba import njit
except ImportError:
    njit = None


if njit is not None:

    @njit(cache=True)
    def _estim_model_numba(ordre, N, coef_autocorr, AI, buff, ring_start):
        coef_reflexion = np.zeros(ordre, dtype=np.float64)

        if coef_autocorr[0] <= 0:
            coef_autocorr[0] = 1.0

        coef_reflexion[0] = -coef_autocorr[1] / coef_autocorr[0]
        AI[0] = 1.0
        AI[1] = coef_reflexion[0]
        variance_erreur_residuelle = (
            coef_autocorr[0] + coef_autocorr[1] * coef_reflexion[0]
        )
        erreur_residuelle = 0.0

        if ordre > 1:
            i_ordre = 1
            while i_ordre < ordre and variance_erreur_residuelle > 0:
                if variance_erreur_residuelle > 0:
                    S = 0.0
                    for i in range(i_ordre):
                        S = S + AI[i] * coef_autocorr[i_ordre - i + 1]

                    coef_reflexion[i_ordre] = -S / variance_erreur_residuelle

                    MH = i_ordre // 2 + 1
                    for i in range(1, MH):
                        IB = i_ordre - i + 2
                        tmp = AI[i] + coef_reflexion[i_ordre] * AI[IB]
                        AI[IB] = AI[IB] + coef_reflexion[i_ordre] * AI[i]
                        AI[i] = tmp
                    AI[i_ordre + 1] = coef_reflexion[i_ordre]
                    variance_erreur_residuelle = (
                        variance_erreur_residuelle + coef_reflexion[i_ordre] * S
                    )

                i_ordre += 1

        if variance_erreur_residuelle > 0:
            variance_erreur_residuelle = variance_erreur_residuelle / float(N - 1)
            erreur_residuelle = 0.0
            for i in range(ordre + 1):
                idx = ring_start + N - i - 1
                if idx >= N:
                    idx -= N
                erreur_residuelle = erreur_residuelle + AI[i] * buff[idx]

        return erreur_residuelle, variance_erreur_residuelle

    @njit(cache=True)
    def _next_rupture_numba(data, start, ordre, Lmin, lamb, biais):
        long_signal = len(data)
        t = start
        rupture = False
        Wn = 0.0
        maxi_value = 0.0
        maxi_t = -1

        size = ordre + 2
        ft = np.zeros(size, dtype=np.float64)
        ftm1 = np.zeros(size, dtype=np.float64)
        variance_f = np.ones(size, dtype=np.float64)
        variance_b = np.ones(size, dtype=np.float64)
        et = np.zeros(size, dtype=np.float64)
        cor = np.zeros(size, dtype=np.float64)

        echantillon = data[t]
        length = 1
        oubli = 1.0 / float(length)
        variance_f[0] = variance_f[0] + oubli * (echantillon**2 - variance_f[0])
        variance_b[0] = variance_f[0]
        et[0] = echantillon
        ft[0] = echantillon
        if ordre < length - 1:
            ik = ordre
        else:
            ik = length - 1
        long_erreur = et[ik]
        long_variance = variance_f[ik]

        court_ready = False
        court_N = Lmin - 1
        if court_N < 1:
            court_N = 1
        buff = np.zeros(court_N, dtype=np.float64)
        ring_start = 0
        coef_autocorr = np.zeros(size, dtype=np.float64)
        AI = np.zeros(size, dtype=np.float64)
        court_erreur = 0.0
        court_variance = 0.0

        while (not rupture) and t < long_signal - 1:
            t += 1
            echantillon = data[t]

            length += 1
            ftm1[:] = ft
            et[0] = echantillon
            oubli = 1.0 / float(length)
            variance_f[0] = variance_f[0] + oubli * (echantillon**2 - variance_f[0])
            variance_b[0] = variance_f[0]
            if ordre < length - 1:
                ik = ordre
            else:
                ik = length - 1

            for n in range(ik + 1):
                oubli = 1.0 / float(length - n)
                cor[n] = cor[n] + oubli * (ftm1[n] * et[n] - cor[n])
                denominateur = variance_f[n] + variance_b[n]
                if denominateur == 0:
                    knplus1 = 2 * cor[n] / np.spacing(variance_f[n])
                else:
                    knplus1 = 2 * cor[n] / denominateur

                et_n = et[n]
                ftm1_n = ftm1[n]
                et[n + 1] = et_n - knplus1 * ftm1_n
                ft[n + 1] = ftm1_n - knplus1 * et_n
                variance_f[n + 1] = variance_f[n + 1] + oubli * (
                    et[n + 1] ** 2 - variance_f[n + 1]
                )
                variance_b[n + 1] = variance_b[n + 1] + oubli * (
                    ft[n + 1] ** 2 - variance_b[n + 1]
                )

            ft[0] = echantillon
            long_erreur = et[ik + 1]
            long_variance = variance_f[ik + 1]

            offset = t - start
            if offset >= Lmin:
                if offset == Lmin:
                    for i in range(court_N):
                        buff[i] = data[start + 1 + i]

                    for tau in range(ordre + 1):
                        coef_tau = coef_autocorr[tau]
                        for i in range(court_N - tau):
                            coef_tau = coef_tau + buff[i] * buff[i + tau - 1]
                        coef_autocorr[tau] = coef_tau

                    court_erreur, court_variance = _estim_model_numba(
                        ordre, court_N, coef_autocorr, AI, buff, ring_start
                    )
                    court_ready = True

                if offset > Lmin and court_ready:
                    dernier_echantillon = buff[ring_start]
                    buff[ring_start] = echantillon
                    ring_start += 1
                    if ring_start == court_N:
                        ring_start = 0
                    dernier_buff_idx = ring_start + court_N - 1
                    if dernier_buff_idx >= court_N:
                        dernier_buff_idx -= court_N
                    dernier_buff = buff[dernier_buff_idx]

                    for tau in range(1, ordre + 1):
                        tau_minus_1_idx = ring_start + tau - 1
                        if tau_minus_1_idx >= court_N:
                            tau_minus_1_idx -= court_N
                        n_minus_tau_minus_1_idx = ring_start + court_N - tau - 1
                        if n_minus_tau_minus_1_idx >= court_N:
                            n_minus_tau_minus_1_idx -= court_N
                        coef_autocorr[tau] = (
                            coef_autocorr[tau]
                            - dernier_echantillon * buff[tau_minus_1_idx]
                            + buff[n_minus_tau_minus_1_idx] * dernier_buff
                        )
                    coef_autocorr[0] = (
                        coef_autocorr[0] - dernier_echantillon**2 + dernier_buff**2
                    )
                    court_erreur, court_variance = _estim_model_numba(
                        ordre, court_N, coef_autocorr, AI, buff, ring_start
                    )

                if court_variance == 0:
                    numerateur = 1e-9
                else:
                    numerateur = court_variance

                QV = numerateur / long_variance
                distance = (
                    2 * court_erreur * long_erreur / long_variance
                    - (1.0 + QV) * long_erreur**2 / long_variance
                    + QV
                    - 1.0
                ) / (2.0 * QV)
                Wn = Wn + distance - biais

                if Wn > maxi_value:
                    maxi_value = Wn
                    maxi_t = t

                if (maxi_value - Wn) > lamb:
                    rupture = True

        return maxi_t, rupture


def printhelp():
    print("Calcul de la segmentation en divergence F/B")
    print("Usage :")
    print(
        "\tdiverg.py -i <audiofile> -b <boundariesOutputFile> [ -o <ordre> ] [-v] [-h] "
    )
    print("\t\t -i : Fichier audio à analyser")
    print("\t\t -o : Ordre de l'analyse. Par défaut : %s" % order)
    print("\t\t -b : Fichier de sortie. Par défaut : %s" % outputPath)
    print("\t\t -h : Affiche ce message")
    exit(1)


class ModelLongTerm(object):
    """
    Modelisation Long-Terme par la méthode d'autocorrelation.
    Initialisé sur les 'Lmin' premiers echantillons depuis la dernière
    rupture.

    Mis à jour echantillon après echantillon, le model grandit au fil
    des itérations.

    """

    def __init__(self, ordre, echantillon):

        self.ordre = ordre
        self.ft = [0] * (ordre + 2)
        self.ftm1 = [0] * (ordre + 2)
        self.variance_f = [1] * (ordre + 2)
        self.variance_b = [1] * (ordre + 2)
        self.et = [0] * (ordre + 2)
        self.cor = [0] * (ordre + 2)
        self.length = 1
        self.erreur_residuelle = 0
        self.variance_erreur_residuelle = 0

        oubli = 1.0 / float(self.length)

        self.variance_f[0] = self.variance_f[0] + oubli * (
            echantillon**2 - self.variance_f[0]
        )
        self.variance_b[0] = self.variance_f[0]
        self.et[0] = echantillon
        self.ft[0] = echantillon

        if ordre < self.length - 1:
            ik = ordre
        else:
            ik = self.length - 1
        self.erreur_residuelle = self.et[ik]
        self.variance_erreur_residuelle = self.variance_f[ik]

    def miseAJour(self, echantillon):
        """
        Mise a jour du model par ajout d'un echantillon.

        """

        self.length += 1
        length = self.length
        ordre = self.ordre
        ft = self.ft
        ftm1 = self.ftm1
        et = self.et
        cor = self.cor
        variance_f = self.variance_f
        variance_b = self.variance_b

        ftm1[:] = ft
        et[0] = echantillon

        oubli = 1.0 / float(length)
        variance_f[0] = variance_f[0] + oubli * (echantillon**2 - variance_f[0])
        variance_b[0] = variance_f[0]
        if ordre < length - 1:
            ik = ordre
        else:
            ik = length - 1

        for n in range(ik + 1):
            oubli = 1.0 / float(length - n)

            cor[n] = cor[n] + oubli * (ftm1[n] * et[n] - cor[n])
            denominateur = variance_f[n] + variance_b[n]
            if denominateur == 0:
                knplus1 = 2 * cor[n] / spacing(variance_f[n])
            else:
                knplus1 = 2 * cor[n] / denominateur

            et_n = et[n]
            ftm1_n = ftm1[n]
            et[n + 1] = et_n - knplus1 * ftm1_n
            ft[n + 1] = ftm1_n - knplus1 * et_n

            variance_f[n + 1] = variance_f[n + 1] + oubli * (
                et[n + 1] ** 2 - variance_f[n + 1]
            )
            variance_b[n + 1] = variance_b[n + 1] + oubli * (
                ft[n + 1] ** 2 - variance_b[n + 1]
            )

        ft[0] = echantillon
        self.erreur_residuelle = et[ik + 1]
        self.variance_erreur_residuelle = variance_f[ik + 1]

    def __str__(self):
        """
        Affichage console.
        """

        s = "Model Long Terme\n"
        s += f"\tOrdre\t\t{self.ordre}\n"
        s += f"\tLongueur\t{self.length}\n"
        s += "\tet\t\t["
        for e in self.et:
            s += f"{e} "
        s += "]\n"
        s += "\tft\t\t["
        for e in self.ft:
            s += f"{e} "
        s += "]\n"
        s += "\tft-1\t\t["
        for e in self.ftm1:
            s += f"{e} "
        s += "]\n"
        s += "\tVarb\t\t["
        for e in self.variance_b:
            s += f"{e} "
        s += "]\n"
        s += "\tVarf\t\t["
        for e in self.variance_f:
            s += f"{e} "
        s += "]\n"
        s += "\tErreur\t\t%f\n" % self.erreur_residuelle
        s += "\tVar(err)\t%f\n" % self.variance_erreur_residuelle
        return s


class ModelCourtTrerm(object):
    """
    Model court terme, glissant de longueur fixe 'Lmin'. en utilisant la
    méthode des treillis.
    """

    def __init__(self, ordre, buff):
        """

        Constructor

        """

        self.N = len(buff)
        self.ordre = ordre
        self.erreur_residuelle = 0
        self.variance_erreur_residuelle = 0
        self.coef_autocorr = [0] * (self.ordre + 2)
        self.AI = [0] * (self.ordre + 2)
        self.dernier_echantillon = 0
        self.buff = buff
        buff_values = list(buff)
        coef_autocorr = self.coef_autocorr
        for tau in range(self.ordre + 1):
            coef_tau = coef_autocorr[tau]
            for i in range(self.N - tau):
                coef_tau = coef_tau + buff_values[i] * buff_values[i + tau - 1]
            coef_autocorr[tau] = coef_tau
        self.estimModel()

    def estimModel(self):

        coef_reflexion = [0] * self.ordre
        coef_autocorr = self.coef_autocorr
        AI = self.AI

        if coef_autocorr[0] <= 0:
            coef_autocorr[0] = 1.0

        coef_reflexion[0] = -coef_autocorr[1] / coef_autocorr[0]
        AI[0] = 1
        AI[1] = coef_reflexion[0]
        self.variance_erreur_residuelle = (
            coef_autocorr[0] + coef_autocorr[1] * coef_reflexion[0]
        )

        if self.ordre > 1:
            i_ordre = 1
            while i_ordre < self.ordre and self.variance_erreur_residuelle > 0:
                if self.variance_erreur_residuelle > 0:
                    S = 0
                    for i in range(i_ordre):
                        S = S + AI[i] * coef_autocorr[i_ordre - i + 1]

                    # coef reflexion
                    coef_reflexion[i_ordre] = -S / self.variance_erreur_residuelle

                    MH = i_ordre // 2 + 1
                    for i in range(1, MH):
                        IB = i_ordre - i + 2
                        tmp = AI[i] + coef_reflexion[i_ordre] * AI[IB]
                        AI[IB] = AI[IB] + coef_reflexion[i_ordre] * AI[i]
                        AI[i] = tmp
                    AI[i_ordre + 1] = coef_reflexion[i_ordre]
                    self.variance_erreur_residuelle = (
                        self.variance_erreur_residuelle + coef_reflexion[i_ordre] * S
                    )

                i_ordre += 1

        if self.variance_erreur_residuelle > 0:
            self.variance_erreur_residuelle = self.variance_erreur_residuelle / float(
                self.N - 1
            )
            self.erreur_residuelle = 0
            buff = self.buff
            N = self.N
            for i in range(self.ordre + 1):
                self.erreur_residuelle = (
                    self.erreur_residuelle + AI[i] * buff[N - i - 1]
                )

    def miseAJour(self, echantillon):
        buff = self.buff
        N = self.N
        coef_autocorr = self.coef_autocorr
        self.dernier_echantillon = buff.popleft()
        dernier_echantillon = self.dernier_echantillon
        buff.append(echantillon)
        dernier_buff = buff[N - 1]
        for tau in range(1, self.ordre + 1):
            coef_autocorr[tau] = (
                coef_autocorr[tau]
                - dernier_echantillon * buff[tau - 1]
                + buff[N - tau - 1] * dernier_buff
            )
        coef_autocorr[0] = coef_autocorr[0] - dernier_echantillon**2 + dernier_buff**2
        self.estimModel()

    def __str__(self):
        """ """
        s = "Model Court Terme\n"
        s += f"\tOrdre\t{self.ordre}\n"
        s += "\tAI\t["
        for e in self.AI:
            s += f"{e} "
        s += "]\n"
        s += f"\tErreur\t{self.erreur_residuelle}\n"
        s += f"\tVar(err)\t{self.variance_erreur_residuelle}\n"
        s += "\tAutocor\t ["
        for e in self.coef_autocorr:
            s += f"{e} "
        s += "]\n"
        return s


def calculDistance(modeleLong, modeleCourt):
    """
    Calcul de la distance entre les modèles longs et court terme

    args :
        - modeleLong (ModelLongTerme) : Modèle appris sur tous les echantillons depuis la dernière rupture
        - modeleCourt (ModelCourtTrerm): Modèle appris sur les Lmin derniers echantillons
    """

    if modeleCourt.variance_erreur_residuelle == 0:
        numerateur = 1e-9
    else:
        numerateur = modeleCourt.variance_erreur_residuelle

    denominateur = modeleLong.variance_erreur_residuelle

    QV = numerateur / denominateur
    return (
        2 * modeleCourt.erreur_residuelle * modeleLong.erreur_residuelle / denominateur
        - (1.0 + QV) * modeleLong.erreur_residuelle**2 / denominateur
        + QV
        - 1.0
    ) / (2.0 * QV)


def _segment_numba_driver(data, fe, ordre, Lmin, lamb, biais, with_backward):
    frontieres = []
    t = 0
    rupt_last = t
    long_signal = len(data)
    Lmin = int(Lmin * fe)

    while t < long_signal - 1:
        t_rupt, rupture = _next_rupture_numba(data, t, ordre, Lmin, lamb, biais)

        if t_rupt > -1:
            m = "forward"
            if with_backward:
                bdata = data[t_rupt:rupt_last:-1]

                if len(bdata) > 0:
                    front = _segment_numba_driver(
                        bdata,
                        fe,
                        ordre,
                        float(Lmin) / fe,
                        lamb,
                        biais,
                        False,
                    )
                    t_bs = [t_rupt - tr for tr, _ in front]

                    if len(t_bs) > 0:
                        t_rupt = t_bs[-1]
                        m = "backward"
        else:
            t_rupt = rupt_last + Lmin
            m = "instable"

        t = t_rupt
        rupt_last = t_rupt

        if rupture:
            frontieres.append((t_rupt, m))

    return frontieres


def segment(data, fe, ordre=2, Lmin=0.02, lamb=40.0, biais=-0.2, with_backward=True):
    """
    Fonction principale de segmentation.

    args :
        - data (list of float): echantillons du signal
        - fe  (float) : fréquence d'échantillonage
        - ordre (int) : ordre des modèles. Par défaut = 2
        - Lmin (float) : longueur minimale d'un segment et taille du buffer pour l'apprentissage du model court terme. Par défaut = 0.02
        - lamb (float) : valeur de lambda pour la détection de chute de Wn. Par défaut = 40.0
        - biais (float) : valeur du bias à appliquer (en négatif). Par défaut = -0.2
        - with_backward (Bool) : Interrupteur du calcul ou non en backward. Par défaut = True
    """
    if njit is not None:
        data_array = np.asarray(data, dtype=np.float64)
        return _segment_numba_driver(
            data_array, fe, ordre, Lmin, lamb, biais, with_backward
        )

    # Initialisation
    frontieres = []
    t = 0
    rupt_last = t
    long_signal = len(data)
    # taille minimum en echantillons
    Lmin = int(Lmin * fe)
    while t < long_signal - 1:
        # Nouvelle Rupture

        #    Critere d'arret : decouverte d'une rupture
        rupture = False

        # Cumulateur de vraissemblance
        Wn = 0.0

        # Valeur et emplacement de la valeur max
        maxi = (0, -1)

        audio_buffer = deque([], Lmin)

        # Initialisation du modèle long terme
        echantillon = data[t]
        longTerme = ModelLongTerm(ordre, echantillon)

        while (not rupture) and t < long_signal - 1:
            t += 1

            # Mise à jour du long terme
            echantillon = data[t]
            longTerme.miseAJour(echantillon)

            # Si l'ecart avec la dernière rupture est suffisant
            # pour utiliser le modèle court terme
            if t - rupt_last >= Lmin:
                # Initialisation du modèle court terme
                if t - rupt_last == Lmin:
                    courtTerme = ModelCourtTrerm(ordre, audio_buffer)

                # Mise à jour du modèle court terme
                if t - rupt_last > Lmin:
                    courtTerme.miseAJour(echantillon)

                # mise à jour du critère
                if courtTerme.variance_erreur_residuelle == 0:
                    numerateur = 1e-9
                else:
                    numerateur = courtTerme.variance_erreur_residuelle

                denominateur = longTerme.variance_erreur_residuelle

                QV = numerateur / denominateur
                distance = (
                    2
                    * courtTerme.erreur_residuelle
                    * longTerme.erreur_residuelle
                    / denominateur
                    - (1.0 + QV) * longTerme.erreur_residuelle**2 / denominateur
                    + QV
                    - 1.0
                ) / (2.0 * QV)
                Wn = Wn + distance - biais

                # Recherche de nouveau maximum
                if Wn > maxi[0]:
                    maxi = (Wn, t)

                # Recherche de rupture par chute superieure à lambda
                if (maxi[0] - Wn) > lamb:
                    rupture = True
            else:
                # Sinon, on prepare l'initialisation
                audio_buffer.append(echantillon)

        # Positionnement de la rupture au dernier point maximum
        t_rupt = maxi[1]

        # Si une rupture à été detecté avec un modèle stable (Wn à croit)
        if t_rupt > -1:
            m = "forward"
            if with_backward:
                bdata = data[t_rupt:rupt_last:-1]

                if len(bdata) > 0:
                    front = segment(
                        bdata,
                        fe,
                        ordre,
                        float(Lmin) / fe,
                        lamb,
                        biais,
                        with_backward=False,
                    )
                    t_bs = [t_rupt - tr for tr, _ in front]

                    if len(t_bs) > 0:
                        t_rupt = t_bs[-1]
                        m = "backward"

        # Sinon on crée un segment de longueur minimale
        else:
            t_rupt = rupt_last + Lmin
            m = "instable"

        # Mise à jour des frontières
        t = t_rupt
        rupt_last = t_rupt

        if rupture:
            frontieres.append((t_rupt, m))

    return frontieres


if __name__ == "__main__":
    #  Lecture du fichier son et passage en float
    try:
        opts, args = getopt(argv[1:], "hi:o:vb:")
    except GetoptError:
        printhelp()

    VERBOSE = False
    order = 16

    for opt, arg in opts:
        if opt == "-h":
            printhelp()
        elif opt == "-i":
            inputPath = arg
        elif opt == "-o":
            order = int(arg)
        elif opt == "-b":
            outputPath = arg
        elif opt == "-v":
            VERBOSE = True

    if not inputPath == None:
        fe, data = wavread(inputPath)
    else:
        printhelp()

    data = [float(i) for i in data]

    if VERBOSE:
        st = time.time()

    frontieres = segment(data, fe, ordre=order, with_backward=True)

    if VERBOSE:
        print(
            "Ordre %d : %d Frontieres (%.4f sec)"
            % (order, len(frontieres), time.time() - st)
        )

    # Sortie texte
    with open(outputPath, "w") as f:
        for t, m in frontieres:
            f.write("%f\t%s\n" % (float(t) / fe, m))
