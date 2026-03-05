import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import NumberInput from '../../components/common/NumberInput';

describe('NumberInput', () => {
  it('renders label and value', () => {
    render(<NumberInput label="Platelets" value={500} onChange={() => {}} />);
    expect(screen.getByText('Platelets')).toBeInTheDocument();
    expect(screen.getByDisplayValue('500')).toBeInTheDocument();
  });

  it('calls onChange with number value', () => {
    const onChange = vi.fn();
    render(<NumberInput label="Count" value={100} onChange={onChange} />);
    fireEvent.change(screen.getByDisplayValue('100'), { target: { value: '200' } });
    expect(onChange).toHaveBeenCalledWith(200);
  });

  it('respects min/max/step attributes', () => {
    render(<NumberInput label="Val" value={5} onChange={() => {}} min={0} max={10} step={0.5} />);
    const input = screen.getByDisplayValue('5') as HTMLInputElement;
    expect(input.min).toBe('0');
    expect(input.max).toBe('10');
    expect(input.step).toBe('0.5');
  });

  it('can be disabled', () => {
    render(<NumberInput label="Val" value={5} onChange={() => {}} disabled />);
    expect(screen.getByDisplayValue('5')).toBeDisabled();
  });
});
