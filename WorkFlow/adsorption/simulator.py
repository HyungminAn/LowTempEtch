class AdsorptionSimulator():
    def __init__(self, inputs):
        self.dst = inputs['dst']

        self.path_slab = inputs['adsorption']['paths']['slab']
        self.path_additive = inputs['adsorption']['paths']['additive']
        self.path_etchant = inputs['adsorption']['paths']['etchant']

        self.additive_name = inputs['adsorption']['mol_info']['additive']['name']
        self.additive_n_layer = inputs['adsorption']['mol_info']['additive']['n_layer']
        self.additive_n_repeat = inputs['adsorption']['mol_info']['additive']['n_repeat']

        self.etchant_name = inputs['adsorption']['mol_info']['etchant']['name']
        self.etchant_n_layer = inputs['adsorption']['mol_info']['etchant']['n_layer']
        self.etchant_n_repeat = inputs['adsorption']['mol_info']['etchant']['n_repeat']

        self.md_flag = inputs['adsorption']['md']['flag']
        self.md_time = inputs['adsorption']['md']['time']
        self.md_temp = inputs['adsorption']['md']['temp']

        self.perturb_flag = inputs['adsorption']['perturb']['flag']
        self.perturb_scale = inputs['adsorption']['perturb']['scale']
        self.perturb_cutoff = inputs['adsorption']['perturb']['cutoff']

        self.fix_h = inputs['constraint']['fix_bottom_height']

    def run(self):
        self._add_additive_to_slab()
        self._add_etchant_to_slab()
        self._summarize_results()

        # make_slab_with_additive(output, **inputs)
        # repeat_adsorption(output, **inputs)
        # summarize_results(output, **inputs)

    def _add_additive_to_slab(self):
        pass

    def _add_etchant_to_slab(self):
        pass

    def _make_slab_with_additive(self):
        pass

    def _repeat_adsorption(self):
        pass

    def _summarize_results(self):
        pass
