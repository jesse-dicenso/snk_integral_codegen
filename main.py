import codegen

# Generate a (D|P) same center VRR kernel
codegen.generate_vrr(MSA="D", MSB="P", same_center=True, dname="SNK_CODEGEN", verbose=True)

# Generate a (G|F) different center HRR kernel
codegen.generate_hrr(MSA="G", MSB="F", same_center=False, dname="SNK_CODEGEN", verbose=False)
