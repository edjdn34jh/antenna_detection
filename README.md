# antenna_detection
Antenna detection from very high resolution optical images.
 Dalles PCRS : https://geoservices.ign.fr/pcrs, jeu de données antennes (pas vérité terrain) : https://data.anfr.fr/visualisation/export/?id=observatoire_2g_3g_4g&disjunctive.adm_lb_nom&refine.adm_lb_nom=BOUYGUES+TELECOM
  détection des antennes sur bâtiment (pas sur pylone ni nature) chacune des composantes de l'antenne (cellule), jeu de données vérité terrain à obtenir, sur haut-de-france par exemple (pour superposition PCRS), un département.
   Quand on a les données de l'anfr, utiliser le code d'axel pour récupérer les zones d'antennes, puis les miens pour tester le modèle yolo avec jeux d'entrainement, validation, test.
