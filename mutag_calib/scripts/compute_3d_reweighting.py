import os
import argparse
from collections import defaultdict
import numpy as np
import hist
import correctionlib
import correctionlib.convert
import rich
from coffea.util import save, load

def dense_axes(h):
    '''Returns the list of dense axes of a histogram.'''
    dense_axes = []
    if type(h) == dict:
        h = h[list(h.keys())[0]]
    for ax in h.axes:
        if not type(ax) in [hist.axis.StrCategory, hist.axis.IntCategory]:
            dense_axes.append(ax)
    return dense_axes

def stack_sum(stack):
    '''Returns the sum histogram of a stack (`hist.stack.Stack`) of histograms.'''
    if len(stack) == 1:
        return stack[0]
    else:
        htot = stack[0]
        for h in stack[1:]:
            htot = htot + h
        return htot

def get_axis_items(h, axis_name):
    axis = h.axes[axis_name]
    return list(axis.value(range(axis.size)))

def get_data_mc_ratio(h_data, h_qcd, h_diff):
    if type(h_data) == hist.Stack:
        h_data = stack_sum(h_data)
    if type(h_qcd) == hist.Stack:
        h_qcd = stack_sum(h_qcd)
    if type(h_diff) == hist.Stack:
        h_diff = stack_sum(h_diff)
    data = h_data.values()
    qcd = h_qcd.values()
    diff = h_diff.values()
    num = data - diff
    den = qcd
    ratio = num / den
    sumw2_num = h_data.variances() + h_diff.variances()
    sumw2_den = h_qcd.variances()
    # Statistical uncertainty on the reweighting SF taking into account
    # the uncertainty on data, QCD, top and WJets
    unc = np.sqrt( sumw2_num/den**2 + (num**2/den**4)*sumw2_den )
    # Statistical uncertainty on the reweighting SF taking into account
    # the uncertainty on data and QCD only
    unc_no_diff = np.sqrt( data/den**2 + (num**2/den**4)*sumw2_den )
    unc[np.isnan(unc)] = 0
    unc_no_diff[np.isnan(unc_no_diff)] = 0

    return ratio, unc, unc_no_diff

def overwrite_check(outfile):
    if os.path.exists(outfile):
        raise Exception(f"Output file {outfile} already exists. Please specify a different output file or run with --overwrite flag to overwrite the output file.")

pt_eta_2d_maps = [
    'FatJetGood_pt_eta',
    #'FatJetGoodNMuon1_pt_eta',
    #'FatJetGoodNMuon2_pt_eta',
    #'FatJetGoodNMuonSJ1_pt_eta',
    #'FatJetGoodNMuonSJUnique1_pt_eta',
]
pt_eta_tau21_3d_maps = [
    'FatJetGood_pt_eta_tau21', 'FatJetGood_pt_eta_tau21_bintau05',
    #'FatJetGoodNMuon1_pt_eta_tau21', 'FatJetGoodNMuon1_pt_eta_tau21_bintau05',
    #'FatJetGoodNMuon2_pt_eta_tau21', 'FatJetGoodNMuon2_pt_eta_tau21_bintau05',
    #'FatJetGoodNMuonSJ1_pt_eta_tau21', 'FatJetGoodNMuonSJ1_pt_eta_tau21_bintau05',
    #'FatJetGoodNMuonSJUnique1_pt_eta_tau21', 'FatJetGoodNMuonSJUnique1_pt_eta_tau21_bintau05',
]

def pt_reweighting(accumulator, histname, output, test=False, overwrite=False):
    years = accumulator["datasets_metadata"]["by_datataking_period"].keys()
    h = accumulator['variables'][histname]
    samples = h.keys()
    samples_data = list(filter(lambda d: 'DATA' in d, samples))
    samples_mc = list(filter(lambda d: 'DATA' not in d, samples))
    samples_qcd = list(filter(lambda d: 'QCD_MuEnriched' in d, samples_mc))
    samples_vjets_top = list(filter(lambda d: (('VJets' in d) | ('SingleTop' in d) | ('TTto4Q' in d)), samples_mc))

    # Compute a 3D correction for each year and save it in a separate json file
    for year in years:
        # Build QCD, VJets+top and Data histograms by summing over all datasets and filtering by year
        h_qcd = sum([h[s][d] for s, datasets_dict in h.items() for d in datasets_dict if ((s in samples_qcd) & (year in d))])
        h_vjets_top = sum([h[s][d] for s, datasets_dict in h.items() for d in datasets_dict if ((s in samples_vjets_top) & (year in d))])
        h_data = sum([h[s][d] for s, datasets_dict in h.items() for d in datasets_dict if ((s in samples_data) & (year in d))])

        axes = dense_axes(h_qcd)
        categories = get_axis_items(h_qcd, 'cat')
        variations = get_axis_items(h_qcd, 'variation')

        ratio_dict = defaultdict(float)

        for cat in categories:
            ratio_dict[cat] = {}
            for var_shape in variations:
                slicing_mc = {'cat': cat, 'variation': var_shape}

                if 'era' in h_data.axes.name:
                    slicing_data = {'cat': cat, 'era': sum}
                else:
                    slicing_data = {'cat': cat}
                ratio, unc, unc_no_diff = get_data_mc_ratio(
                    h_data[slicing_data],
                    h_qcd[slicing_mc],
                    h_vjets_top[slicing_mc]
                )
                mod_ratio  = np.nan_to_num(ratio, nan=1.0)
                mod_unc = np.nan_to_num(unc, nan=0.0)
                mod_unc_no_diff = np.nan_to_num(unc_no_diff, nan=0.0)

                ratio_dict[cat][var_shape] = {}
                ratio_dict[cat][var_shape].update({ "nominal" : mod_ratio })
                ratio_dict[cat][var_shape].update({ "statUp" : mod_ratio + mod_unc })
                ratio_dict[cat][var_shape].update({ "statDown" : mod_ratio - mod_unc })

        categories = list(ratio_dict.keys())
        shape_variations = list(ratio_dict[categories[0]].keys())
        variations = list(ratio_dict[categories[0]][shape_variations[0]].keys())
        axis_category = hist.axis.StrCategory(categories, name="cat")
        axis_shape_variation = hist.axis.StrCategory(shape_variations, name="shape_variation")
        axis_variation = hist.axis.StrCategory(variations, name="variation")
        # Stack nominal, statUp, statDown maps for each category
        stack_map = np.stack([[list(ratio_dict[cat][var_shape].values()) for var_shape in shape_variations] for cat in categories])
        sfhist = hist.Hist(axis_category, axis_shape_variation, axis_variation, *axes, data=stack_map)
        sfhist.label = "out"
        sfhist.name = f"{histname}_corr_{year}"
        description = "Reweighting SF matching the leading fatjet pT and eta MC distribution to data."
        clibcorr = correctionlib.convert.from_histogram(sfhist, flow="clamp")
        clibcorr.description = description
        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description="MC to data reweighting SF",
            corrections=[clibcorr],
        )
        rich.print(cset)

        os.makedirs(output, exist_ok=True)
        outfile_reweighting = os.path.join(output, f'{histname}_{year}_reweighting.json')
        if not overwrite:
            overwrite_check(outfile_reweighting)
        print(f"Saving pt reweighting factors in {outfile_reweighting}")
        with open(outfile_reweighting, "w") as fout:
            fout.write(cset.model_dump_json(exclude_unset=True))
        fout.close()
        if test:
            print(f"Loading correction from {outfile_reweighting}")
            cset = correctionlib.CorrectionSet.from_file(os.path.abspath(outfile_reweighting))
            cat = categories[0]
            print(f"(cat): {cat},", "(var): nominal")
            pt_corr = cset[sfhist.name]
            pos  = np.array([0, 1, 0, 1, 0], dtype=int)
            pt  = np.array([50, 100, 400, 500, 1000], dtype=float)
            eta = np.array([-2, -1, 0, 1, 2], dtype=float)
            print("pos =", pos)
            print("pt =", pt)
            print("eta =", eta)
            if histname in pt_eta_2d_maps:
                args = (pos, pt, eta)

            elif histname in pt_eta_tau21_3d_maps:
                tau21 = np.array([0.1, 0.35, 0.65, 0.8, 0.9], dtype=float)
                print("tau21 =", tau21)
                args = (pos, pt, eta, tau21)
            for var_shape in shape_variations:
                categorical_args = [cat, var_shape, 'nominal']
                print(categorical_args)
                print(pt_corr.evaluate(*categorical_args, *args))
                print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pt reweighting factors.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input coffea file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output directory.")
    parser.add_argument('--test', action='store_true', help="Printout reweighting factors for testing.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite output file.")
    args = parser.parse_args()

    accumulator = load(args.input)

    for histname in pt_eta_2d_maps + pt_eta_tau21_3d_maps:
        pt_reweighting(accumulator=accumulator, histname=histname, output=args.output, test=args.test, overwrite=args.overwrite)
